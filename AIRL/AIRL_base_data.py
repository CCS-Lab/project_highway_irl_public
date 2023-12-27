import gym
import numpy as np
#import tensorflow as tf
from AIRL.policy_net_continuous_discrete import Policy_net
from AIRL.ppo_combo import PPOTrain
from AIRL.interact_with_highway import AIRL_test_function_gym
# from AIRL.AIRL_net_discriminator_blend import Discriminator
import ray
import os
import time
from shared.boost_action import boost_action
from shared.boost_action import polish_action

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# from env_AIRL import highwayEnv_AIRL
# import highway_irl
from random import randrange
import highway_irl_v2

def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def AIRL_base(args,expert_reward):
    
    try:
        state_only = args.state_only
        print("state_only loaded",state_only)
    except:
        state_only = False
        
    if state_only:
        from AIRL.AIRL_state_only_discriminator import Discriminator
    else:
        from AIRL.AIRL_net_discriminator_blend import Discriminator
    
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
    #expert_traj_dir = args.expert_traj_dir + args.envs_1 + '/'
    reward_save_dir = args.reward_savedir + args.envs_1 + '/'
    check_and_create_dir(model_save_dir)
    check_and_create_dir(reward_save_dir)

    if args.continue_s:
        args = np.load(model_save_dir+"setup.npy", allow_pickle=True).item()
        
    env = gym.make('IRL-v2')

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
                   lr_value=args.lr_value, clip_value=args.clip_value)
    #saver = tf.train.Saver(max_to_keep=50)
    
    expert_observations = np.load(expert_traj_dir+'observations_back.npy')

    expert_actions = np.load(expert_traj_dir + 'actions.npy')
    
    if args.boost_action:
        lane_list = expert_observations[:,1]
        expert_actions = boost_action(expert_actions,2,lane_list)
            
    if args.polish_action:
        expert_actions = polish_action(expert_actions,expert_observations)   
    
    #print(expert_actions[:100])
    
    if args.random_starting:        
        #random starting point
        used_traj_length = args.traj_length
        start_point = randrange(args.full_traj_length-used_traj_length+1) #start point for the sampling from expert trajectory
    else:
        start_point = 0 #fixed starting point
        
    # print("start",start_point)
    expert_observations = expert_observations[start_point:args.traj_length+start_point]
    expert_actions = expert_actions[start_point:args.traj_length+start_point]
    
    if args.cv:
        # cv_length = int(args.traj_length/args.fold)
        # print("cv length", cv_length, "cv number", args.sim_number)        
        test_sample = np.arange(args.cv_length) + (args.sim_number-1) * args.cv_length
        print("test range",test_sample[0],test_sample[-1])
        expert_observations = np.delete(expert_observations,test_sample,axis=0)
        expert_actions = np.delete(expert_actions,test_sample)
    
    print(np.shape(expert_observations),np.shape(expert_actions))
    
    if not discrete_env_check:
        act_dim = env.action_space.shape[0]
        expert_actions = np.reshape(expert_actions, [-1, act_dim])
    else:
        expert_actions = expert_actions.astype(np.int32)

    # print(np.shape(expert_actions))

    discrim_ratio = int(np.floor(args.num_expert_dimension / args.min_length))
    discrim_batch_number = args.num_expert_dimension / args.batch_size_discrim

    D = Discriminator('AIRL_discriminator', env, args.lr_discrim, discrim_batch_number, n_feature = args.n_feature, n_units = args.units_d)

    origin_reward_recorder = []
    AIRL_reward_recorder = []
    discrete_reward_recorder = []
    acc_recorder = []
    counter_d = 0
    
    saver = tf.train.Saver(max_to_keep=2)
    best_saver = tf.train.Saver(max_to_keep=1)
    best_saver_d = tf.train.Saver(max_to_keep=1)
    best_saver_acc = tf.train.Saver(max_to_keep=1)
    best_saver_tacc = tf.train.Saver(max_to_keep=1)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())

        if args.continue_s:
            saver.restore(sess, model_save_dir + args.model_restore)
            origin_reward_recorder = np.load(reward_save_dir + "origin_reward.npy").tolist()
            AIRL_reward_recorder = np.load(reward_save_dir + "airl_reward.npy").tolist()
            discrete_reward_recorder = np.load(reward_save_dir + "discrete_reward.npy").tolist()
            try:
                acc_recorder = np.load(reward_save_dir + "acc.npy").tolist()
            except:
                print("no existing acc file to load")

            with open(model_save_dir + args.log_file, 'a+') as r_file:
                r_file.write(
                    "the continue point: {}, the lr_policy: {}, the lr_value: {}, the lr_discrim: {} \n".format(
                        len(origin_reward_recorder), args.lr_policy, args.lr_value, args.lr_discrim))
        
        elif args.prior_model:
            saver.restore(sess, args.model_restore)
            np.save(model_save_dir+"setup.npy", args)
            print("loaded prior model", args.model_restore)
        else:
            np.save(model_save_dir+"setup.npy", args)
        
        min_deviation = 9999
        min_deviation_discrete = 9999
        max_acc = 0
        max_tacc = 0
        best_iters = np.zeros(4)
        
        for iteration in range(args.iteration):
            
            start_time = time.time()

            policy_value = sess.run(Policy.get_trainable_variables())
            discriminator_value = sess.run(D.get_trainable_variables())

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
            
            print(np.shape(action_batch_total))
            
            observation_batch_total = np.array([observation_batch for observation_batch in observation_batch_total])
            action_batch_total = np.array([action_batch for action_batch in action_batch_total])
            rtg_batch_total = np.array([rtg_batch for rtg_batch in rtg_batch_total])

            print("vector_action",np.shape(action_batch_total))
            
            gaes_batch_total = np.array([gaes_batch for gaes_batch in gaes_batch_total])
            value_next_batch_total = np.array([value_next_batch for value_next_batch in value_next_batch_total])
            reward_batch_total = np.array([reward_batch for reward_batch in reward_batch_total])

            gaes_batch_total = (gaes_batch_total - np.mean(gaes_batch_total)) / (
                    np.std(gaes_batch_total) + 1e-10)

            counter_d += 1

            if counter_d >= 0:

                counter_d = 0
                expert_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                     Policy.acts: expert_actions})
                agent_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: observation_batch_total,
                                                                    Policy.acts: action_batch_total})
                
                if state_only:
                    # expert_next_obs = expert_observations[1:]
                    # agent_next_obs = observation_batch_total[1:]                    
                    discrim_batch_expert = [expert_observations[:-1], expert_actions[:-1], expert_sa_ph[:-1],  expert_observations[1:]]
                    discrim_batch_agent = [observation_batch_total[:-1], action_batch_total[:-1], agent_sa_ph[:-1], observation_batch_total[1:]]
                else:
                    discrim_batch_expert = [expert_observations, expert_actions, expert_sa_ph]
                    discrim_batch_agent = [observation_batch_total, action_batch_total, agent_sa_ph]

                for epoch_discrim in range(args.num_epoch_discrim):
                    
                    if state_only:
                        total_index_agent = np.arange(args.min_length - 1) #drop the last index without next_obs
                        total_index_expert = np.arange((args.min_length * discrim_ratio) - 1)
                    else:
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
                            
                        if state_only:
                            # print(sampled_batch_expert[0].shape)
                            # print(sampled_batch_expert[1].shape)
                            # print("state only training",state_only)
                            sampled_obs = np.concatenate((sampled_batch_expert[0],sampled_batch_agent[0]),axis=0)
                            sampled_next_obs = np.concatenate((sampled_batch_expert[3],sampled_batch_agent[3]),axis=0)
                            sampled_prob = np.concatenate((sampled_batch_expert[2],sampled_batch_agent[2]),axis=0)
                            # print(sampled_obs.shape,sampled_prob.shape)
                            expert_label = np.ones(sampled_batch_expert[1].shape)
                            agent_label = np.zeros(sampled_batch_agent[1].shape)
                            sample_label = np.concatenate((expert_label,agent_label))
                            # print(sample_label[:,None].shape)
                            
                            D.train_state(obs_t = sampled_obs,
                                         nobs_t = sampled_next_obs,
                                         lprobs = sampled_prob[:,None],
                                         labels = sample_label[:,None])                            
                        else:
                            D.train(expert_s=sampled_batch_expert[0],
                                    expert_a=sampled_batch_expert[1],
                                    agent_s=sampled_batch_agent[0],
                                    agent_a=sampled_batch_agent[1],
                                    expert_sa_p=sampled_batch_expert[2],
                                    agent_sa_p=sampled_batch_agent[2]
                                    )
                        #print("D updated")
                        #var = [v for v in tf.trainable_variables() if v.name == "AIRL_discriminator/network/prob/bias:0"]
                        #print(sess.run(var))

            print("at {}, the average episode reward is: {}".format(iteration, evaluation_1))
            print("at {}, the average episode AIRL reward is: {}".format(iteration, evaluation_AIRL))
            print("at {}, the average episode discrete reward is: {}".format(iteration, evaluation_discrete))
            print("at {}, elapsed time is: {}".format(iteration, time.time()-start_time))
            print("state only training",state_only)
                
            origin_reward_recorder.append(evaluation_1)
            AIRL_reward_recorder.append(evaluation_AIRL)
            discrete_reward_recorder.append(evaluation_discrete)
            
            ################ action prediction #########################
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
            total_acc = np.mean(expert_actions == action_pred)
            print("total acc", total_acc)
            acc_recorder.append(total_acc)
            ##############################################################
            
            if iteration % 5 == 0 and iteration > 0:
                np.save(reward_save_dir + "origin_reward.npy", origin_reward_recorder)
                np.save(reward_save_dir + "airl_reward.npy", AIRL_reward_recorder)
                np.save(reward_save_dir + "discrete_reward.npy", discrete_reward_recorder)
                np.save(reward_save_dir + "acc.npy", acc_recorder)
                saver.save(sess, model_save_dir + '{}'.format(iteration) + args.model_save)
            
            if iteration > 50:                
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
                    
#                 ### action prediction
#                 est_action_prob = np.zeros((len(expert_actions),5))
#                 for iAction in range(5):
#                     est_action_prob[:,iAction] = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
#                                                                         Policy.acts: np.ones(len(expert_actions))*iAction})
#                 pred_match_list = np.zeros(env.action_space.n)
#                 action_pred = np.squeeze(np.argmax(est_action_prob,axis=1))
                
#                 ### calculate accuracy
#                 for iAction in range(5):
#                     pred_match_list[iAction] = np.mean(expert_actions[expert_actions==iAction]==action_pred[expert_actions==iAction])
#                 normalized_acc = np.mean(pred_match_list)
#                 print("normalized acc", normalized_acc)
                if normalized_acc > max_acc: ## normalized acc or total_acc
                    max_acc = normalized_acc.copy()
                    best_saver_acc.save(sess, model_save_dir + 'best_acc_'+ args.model_save)
                    print("best acc model saved at iteration",iteration)
                    best_iters[2] = iteration
                
                if total_acc > max_tacc: ## normalized acc or total_acc
                    max_tacc = total_acc.copy()
                    best_saver_tacc.save(sess, model_save_dir + 'best_tacc_'+ args.model_save)
                    print("best total acc model saved at iteration",iteration)
                    best_iters[3] = iteration
                
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
                        PPO.train_value_v(obs=sampled_inp_batch[0], v_preds_next=sampled_inp_batch[4],
                                          rewards=sampled_inp_batch[5])
            else:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]

                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]

                        PPO.train_value(obs=sampled_inp_batch[0], rtg=sampled_inp_batch[3])

            for epoch in range(args.num_epoch_policy):
                total_index = np.arange(args.min_length)
                np.random.shuffle(total_index)
                for i in range(0, args.min_length, args.batch_size):
                    sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                    sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                    PPO.train_policy(obs=sampled_inp_batch[0], actions=sampled_inp_batch[1], gaes=sampled_inp_batch[2])