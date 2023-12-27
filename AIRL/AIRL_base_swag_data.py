import gym
import numpy as np
#import tensorflow as tf
from AIRL.policy_net_continuous_discrete import Policy_net
from AIRL.ppo_combo import PPOTrain
from AIRL.interact_with_highway import AIRL_test_function_gym
from AIRL.AIRL_net_discriminator_blend import Discriminator
import ray
import os
import time
from shared.boost_action import boost_action
from shared.boost_action import polish_action
from shared.diminish_lr import diminish_lr
from shared.IRL_analyser import IRL_analyser
from shared.get_reward_function import get_reward_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# from env_AIRL import highwayEnv_AIRL
# import highway_irl
from random import randrange

def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def flatten_param(weights):
    vector_size = [np.product(weights[v].shape) for v in range(len(weights))]
    flat_list = [weights[v].reshape(vector_size[v]) for v in range(len(weights))]
    
    return np.concatenate(flat_list)

def unflatten_param(weights,flat_weights):
    vector_size = [np.product(weights[v].shape) for v in range(len(weights))]
    weight_shape = [weights[v].shape for v in range(len(weights))]
    
    weight_list = []
    for i in range(len(weights)):
        weight_list.append(flat_weights[:vector_size[i]].reshape(weight_shape[i]))
        flat_weights = np.delete(flat_weights,range(vector_size[i]))
    print("remaining elements",len(flat_weights))
    return weight_list

def AIRL_base_swag(args,expert_reward):
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    tf.reset_default_graph()
    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )

    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )

    model_save_dir = args.savedir + args.envs_1 + '/'
    expert_traj_dir = args.expert_traj_dir + '/'
    # expert_traj_dir = args.expert_traj_dir + args.envs_base + '/'
    #expert_traj_dir = args.expert_traj_dir + args.envs_1 + '/'
    reward_save_dir = args.reward_savedir + args.envs_1 + '/'
    check_and_create_dir(model_save_dir)
    check_and_create_dir(reward_save_dir)

    if args.continue_s:
        args = np.load(model_save_dir+"setup.npy", allow_pickle=True).item()

    if args.version == 1:
        import highway_irl
        env = gym.make('IRL-v1')       
        # env_discrete = gym.make('IRL-v1')
    elif args.version == 2:
        import highway_irl_v2
        env = gym.make('IRL-v2')       
        # env_discrete = gym.make('IRL-v2')
            
    #env = gym.make(args.envs_1)
    env.config=np.load('env_configure/'+args.config, allow_pickle = True).tolist()
    print("config loaded:",args.config)
    env.reset()
    
    #print(env.observation_space.shape)

    discrete_env_check = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    env.seed(0)

    if not discrete_env_check:
        print(env.action_space.low)
        print(env.action_space.high)

    Policy = Policy_net('policy', env, args.units_p, args.units_v, n_feature = args.n_feature)
    Old_Policy = Policy_net('old_policy', env, args.units_p, args.units_v, n_feature = args.n_feature)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, lambda_1=args.lambda_1, lr_policy=args.lr_policy,
                   lr_value=args.lr_value, clip_value=args.clip_value, optim = args.optim, swag = args.swag)
    #saver = tf.train.Saver(max_to_keep=50)
    
    if args.two_distance:
        expert_observations = np.load(expert_traj_dir+'observations_double.npy')
    elif args.back_distance:
        expert_observations = np.load(expert_traj_dir+'observations_back.npy')
    else:
        expert_observations = np.load(expert_traj_dir+'observations.npy')
    expert_actions = np.load(expert_traj_dir + 'actions.npy')
    
    if args.boost_action:
        lane_list = expert_observations[:,1]
        if args.version == 1:
            expert_actions = boost_action(expert_actions,5,lane_list)
        elif args.version == 2:
            expert_actions = boost_action(expert_actions,2,lane_list)
    
    if args.polish_action:
        expert_actions = polish_action(expert_actions,expert_observations)
        
    #expert_observations = expert_observations[:args.traj_length]
    #expert_actions = expert_actions[:args.traj_length]
    
    if args.random_starting:        
        #random starting point
        used_traj_length = args.traj_length
        start_point = randrange(args.full_traj_length-used_traj_length+1) #start point for the sampling from expert trajectory
    else:
        start_point = 0 #fixed starting point
        
    # print("start",start_point)
    
    expert_observations = expert_observations[start_point:args.traj_length+start_point]
    expert_actions = expert_actions[start_point:args.traj_length+start_point]
    
    print(np.shape(expert_observations),np.shape(expert_actions))
    
    if not discrete_env_check:
        act_dim = env.action_space.shape[0]
        expert_actions = np.reshape(expert_actions, [-1, act_dim])
    else:
        expert_actions = expert_actions.astype(np.int32)

    print(np.shape(expert_actions))

    discrim_ratio = int(np.floor(args.num_expert_dimension / args.min_length))
    discrim_batch_number = args.num_expert_dimension / args.batch_size_discrim

    D = Discriminator('AIRL_discriminator', env, args.lr_discrim, discrim_batch_number, n_feature = args.n_feature, n_units = args.units_d, optim = args.optim, swag = args.swag)

    origin_reward_recorder = []
    AIRL_reward_recorder = []
    discrete_reward_recorder = []
    counter_d = 0
    
    saver = tf.train.Saver(max_to_keep=25)
    best_saver = tf.train.Saver(max_to_keep=1)
    best_saver_d = tf.train.Saver(max_to_keep=1)
    best_saver_acc = tf.train.Saver(max_to_keep=1)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())

        if args.continue_s:
            saver.restore(sess, model_save_dir + args.model_restore)
            origin_reward_recorder = np.load(reward_save_dir + "origin_reward.npy").tolist()
            AIRL_reward_recorder = np.load(reward_save_dir + "airl_reward.npy").tolist()

            with open(model_save_dir + args.log_file, 'a+') as r_file:
                r_file.write(
                    "the continue point: {}, the lr_policy: {}, the lr_value: {}, the lr_discrim: {} \n".format(
                        len(origin_reward_recorder), args.lr_policy, args.lr_value, args.lr_discrim))

        else:
            np.save(model_save_dir+"setup.npy", args)

        min_deviation = 9999
        min_deviation_discrete = 9999
        max_acc = 0
        best_iters = np.zeros(3)

        for iteration in range(args.iteration):
            
            start_time = time.time()

            policy_value = sess.run(Policy.get_trainable_variables())
            discriminator_value = sess.run(D.get_trainable_variables())

            ####################### swag-update #########################
                        
            lr_policy = diminish_lr(args.lr_policy,args.swag_policy,args.lr_start_iter,args.SWAG_START_ITER,iteration)
            lr_value = diminish_lr(args.lr_value,args.swag_value,args.lr_start_iter,args.SWAG_START_ITER,iteration)
            lr_discrim = diminish_lr(args.lr_discrim,args.swag_discrim,args.lr_start_iter,args.SWAG_START_ITER,iteration)
            
            print("current lr",lr_policy,lr_value,lr_discrim)

            K_SWAG = args.K_SWAG
            # SWAG_START_ITER = 1000
            
            if iteration == args.SWAG_START_ITER:
                # Initialization of SWAG-parameters
                n_SWAG = 1
                first_moment = flatten_param(discriminator_value)
                second_moment = first_moment ** 2
                Dev = np.empty((first_moment.shape[0], 0))  # deviation matrix
                
                new_Dev_col = second_moment - first_moment ** 2
                Dev = np.append(Dev, new_Dev_col.reshape(new_Dev_col.shape[0], 1), axis=1)
                # global_step = 0  # total number of iterations (across different epochs)
                
                # policy values
                first_moment_p = flatten_param(policy_value)
                second_moment_p = first_moment_p ** 2
                P = np.empty((first_moment_p.shape[0], 0))
                
                new_P_col = second_moment_p - first_moment_p ** 2
                P = np.append(P, new_P_col.reshape(new_P_col.shape[0], 1), axis=1)
                
            if iteration > args.SWAG_START_ITER:
                
                new_weights = flatten_param(discriminator_value)
                first_moment = (n_SWAG * first_moment + new_weights) / (n_SWAG + 1)
                second_moment = (n_SWAG * second_moment + new_weights ** 2) / (n_SWAG + 1)
                if Dev.shape[1] == K_SWAG:
                    Dev = np.delete(Dev, 0, 1)  # remove first column
                new_Dev_col = second_moment - first_moment ** 2
                Dev = np.append(Dev, new_Dev_col.reshape(new_Dev_col.shape[0], 1), axis=1)
                
                new_weights_p = flatten_param(policy_value)                
                first_moment_p = (n_SWAG * first_moment_p + new_weights_p) / (n_SWAG + 1)
                second_moment_p = (n_SWAG * second_moment_p + new_weights_p ** 2) / (n_SWAG + 1)
                if P.shape[1] == K_SWAG:
                    P = np.delete(P, 0, 1)  # remove first column
                new_P_col = second_moment_p - first_moment_p ** 2
                P = np.append(P, new_P_col.reshape(new_P_col.shape[0], 1), axis=1)
                
                n_SWAG += 1
            #################### finished swag ############################

            environment_sampling = []

            for i in range(args.num_parallel_sampler):
                x1 = AIRL_test_function_gym.remote(args, policy_value, discriminator_value, discrete_env_check,
                                                   np.ceil(args.min_length / args.num_parallel_sampler), i,
                                                   discrim_batch_number)
                environment_sampling.append(x1)

            results = ray.get(environment_sampling)
            print(np.shape(results))

            sampling_unpack = np.concatenate([result[0] for result in results], axis=1)
            evaluation_1 = np.mean([result[1] for result in results])
            evaluation_AIRL = np.mean([result[2] for result in results])
            evaluation_discrete = np.mean([result[3] for result in results])

            observation_batch_total, action_batch_total, rtg_batch_total, gaes_batch_total, \
            value_next_batch_total, reward_batch_total = sampling_unpack
            
            #print(np.shape(action_batch_total))
                
            observation_batch_total = np.array([observation_batch for observation_batch in observation_batch_total])
            action_batch_total = np.array([action_batch for action_batch in action_batch_total])
            rtg_batch_total = np.array([rtg_batch for rtg_batch in rtg_batch_total])

            gaes_batch_total = np.array([gaes_batch for gaes_batch in gaes_batch_total])
            value_next_batch_total = np.array([value_next_batch for value_next_batch in value_next_batch_total])
            reward_batch_total = np.array([reward_batch for reward_batch in reward_batch_total])

            gaes_batch_total = (gaes_batch_total - np.mean(gaes_batch_total)) / (
                    np.std(gaes_batch_total) + 1e-10)

            counter_d += 1

#             if counter_d >= 2 + (iteration / 500) * 50 or iteration == 0:
#                 print("D updated")
            if counter_d >= 0:

                counter_d = 0
                expert_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                     Policy.acts: expert_actions})
                agent_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: observation_batch_total,
                                                                    Policy.acts: action_batch_total})

                discrim_batch_expert = [expert_observations, expert_actions, expert_sa_ph]
                discrim_batch_agent = [observation_batch_total, action_batch_total, agent_sa_ph]

                for epoch_discrim in range(args.num_epoch_discrim):

                    total_index_agent = np.arange(args.min_length)
                    total_index_expert = np.arange(args.min_length * discrim_ratio)

                    np.random.shuffle(total_index_agent)
                    np.random.shuffle(total_index_expert)

                    for i in range(0, args.min_length, args.batch_size_discrim):
                        sample_indices_agent = total_index_agent[i:min(i + args.batch_size_discrim, args.min_length)]
                        sample_indices_expert = total_index_expert[i * discrim_ratio:min(
                            i * discrim_ratio + args.batch_size_discrim * discrim_ratio,
                            args.min_length * discrim_ratio)]
                        
                        #print(sample_indices_expert)

                        sampled_batch_agent = [np.take(a=a, indices=sample_indices_agent, axis=0) for a in
                                               discrim_batch_agent]
                        sampled_batch_expert = [np.take(a=a, indices=sample_indices_expert, axis=0) for a in
                                                discrim_batch_expert]

                        D.train_swag(expert_s=sampled_batch_expert[0],
                                expert_a=sampled_batch_expert[1],
                                agent_s=sampled_batch_agent[0],
                                agent_a=sampled_batch_agent[1],
                                expert_sa_p=sampled_batch_expert[2],
                                agent_sa_p=sampled_batch_agent[2],
                                lr = lr_discrim
                                )
                        #print("D updated")
                        #var = [v for v in tf.trainable_variables() if v.name == "AIRL_discriminator/network/prob/bias:0"]
                        #print(sess.run(var))

            print("at {}, the average episode reward is: {}".format(iteration, evaluation_1))
            print("at {}, the average episode AIRL reward is: {}".format(iteration, evaluation_AIRL))
            print("at {}, the average episode discrete reward is: {}".format(iteration, evaluation_discrete))
            print("at {}, elapsed time is: {}".format(iteration, time.time()-start_time))

            origin_reward_recorder.append(evaluation_1)
            AIRL_reward_recorder.append(evaluation_AIRL)
            discrete_reward_recorder.append(evaluation_discrete)

            if iteration % 5 == 0 and iteration > 0:
                np.save(reward_save_dir + "origin_reward.npy", origin_reward_recorder)
                np.save(reward_save_dir + "airl_reward.npy", AIRL_reward_recorder)
                np.save(reward_save_dir + "discrete_reward.npy", discrete_reward_recorder)
                saver.save(sess, model_save_dir + '{}'.format(iteration) + args.model_save)

            if iteration > 100:
                if abs(evaluation_1-expert_reward) < min_deviation:
                    min_deviation = abs(evaluation_1-expert_reward)
                    best_saver.save(sess, model_save_dir + 'best_'+ args.model_save)
                    print("best model saved at iteration",iteration)
                    best_iters[0] = iteration

                if abs(evaluation_discrete-expert_reward) < min_deviation_discrete:
                    min_deviation_discrete = abs(evaluation_discrete-expert_reward)
                    best_saver_d.save(sess, model_save_dir + 'best_discrete_'+ args.model_save)
                    print("best discrete model saved at iteration",iteration)
                    best_iters[1] = iteration
                
                ### action prediction
                est_action_prob = np.zeros((len(expert_actions),5))                
                for iAction in range(5):
                    est_action_prob[:,iAction] = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                        Policy.acts: np.ones(len(expert_actions))*iAction})
                pred_match_list = np.zeros(env.action_space.n)
                action_pred = np.squeeze(np.argmax(est_action_prob,axis=1))
                
                ### calculate accuracy
                for iAction in range(5):
                    pred_match_list[iAction] = np.mean(expert_actions[expert_actions==iAction]==action_pred[expert_actions==iAction])
                normalized_acc = np.mean(pred_match_list)
                print("normalized acc", normalized_acc)
                if normalized_acc > max_acc:
                    max_acc = normalized_acc.copy()
                    best_saver_acc.save(sess, model_save_dir + 'best_acc_'+ args.model_save)
                    print("best acc model saved at iteration",iteration)
                    best_iters[2] = iteration
                np.save(model_save_dir+"best_iters.npy",best_iters)

            inp_batch = [observation_batch_total, action_batch_total, gaes_batch_total, rtg_batch_total,
                         value_next_batch_total, reward_batch_total]

            PPO.assign_policy_parameters()

            # train
            if args.alter_value:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                        PPO.train_value_v_swag(obs=sampled_inp_batch[0], v_preds_next=sampled_inp_batch[4],
                                          rewards=sampled_inp_batch[5],lr_value=lr_value)
            else:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]

                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]

                        PPO.train_value_swag(obs=sampled_inp_batch[0], rtg=sampled_inp_batch[3],lr_value=lr_value)

            for epoch in range(args.num_epoch_policy):
                total_index = np.arange(args.min_length)
                np.random.shuffle(total_index)
                for i in range(0, args.min_length, args.batch_size):
                    sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                    sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                    PPO.train_policy_swag(obs=sampled_inp_batch[0], actions=sampled_inp_batch[1], gaes=sampled_inp_batch[2],lr_policy=lr_policy)
            
            if iteration > args.SWAG_START_ITER:
                # Compute SWAG parameters
                param_dict = {}
                param_dict["theta_SWA"] = first_moment
                param_dict["sigma_SWAG"] = second_moment - first_moment ** 2  # NOTE: stored as vector for efficiency
                param_dict["D_SWAG"] = Dev

                param_dict["theta_p_SWA"] = first_moment_p
                param_dict["sigma_p_SWAG"] = second_moment_p - first_moment_p ** 2  # NOTE: stored as vector for efficiency
                param_dict["P_SWAG"] = P

                param_dict["K_SWAG"] = K_SWAG

                np.savez("swag_params/"+args.envs_1+".npz", param_dict)