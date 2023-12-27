import numpy as np
#import tensorflow as tf
from AIRL.policy_net_continuous_discrete import Policy_net
from AIRL.ppo_combo import PPOTrain
# from AIRL.interact_with_highway import AIRL_test_function_gym
# from AIRL.AIRL_net_discriminator_blend import Discriminator
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from env_AIRL import highwayEnv_AIRL
# import highway_irl
import os
import gym

# def check_and_create_dir(dir_path):
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
        
def get_reward_function(args, expert_observations, expert_actions):
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    
    tf.reset_default_graph()
    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )

    try:
        state_only = args.state_only
        print("state_only loaded",state_only)
    except:
        state_only = False
        
    try:
        weighted_y = args.weighted_y
        print("weighted_y loaded",weighted_y)
    except:
        weighted_y = False
        
    if state_only:
        from AIRL.AIRL_state_only_discriminator import Discriminator
    else:
        from AIRL.AIRL_net_discriminator_blend import Discriminator
        
    model_save_dir = args.savedir + args.envs_1 + '/'
    # expert_traj_dir = args.expert_traj_dir + args.envs_1 + '/'
    # reward_save_dir = args.reward_savedir + args.envs_1 + '/'

    if args.version == 1:
        import highway_irl
        env = gym.make('IRL-v1')
    elif args.version == 2:
        import highway_irl_v2
        env = gym.make('IRL-v2')

    action_prob_list = np.zeros((len(expert_actions),5))
        
    # env.config=np.load('env_configure/config_35.npy', allow_pickle = True).tolist()
    env.config=np.load('env_configure/'+args.config, allow_pickle = True).tolist()
    env.reset()

    try:
        optim = args.optim
        print("optimizer:", optim)
    except:
        print("no optimizer information loaded")
        optim = 'Adam'
        
    Policy = Policy_net('policy', env, args.units_p, args.units_v, n_feature = args.n_feature)
    Old_Policy = Policy_net('old_policy', env, args.units_p, args.units_v, n_feature = args.n_feature)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, lambda_1=args.lambda_1, lr_policy=args.lr_policy,
                   lr_value=args.lr_value, clip_value=args.clip_value, optim = optim)
    #saver = tf.train.Saver(max_to_keep=50)

    discrim_ratio = int(np.floor(args.num_expert_dimension / args.min_length))
    discrim_batch_number = args.num_expert_dimension / args.batch_size_discrim

    D = Discriminator('AIRL_discriminator', env, args.lr_discrim, discrim_batch_number, n_feature = args.n_feature, n_units = args.units_d, optim = optim)
    saver = tf.train.Saver(max_to_keep=50)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_dir + args.model_restore)        
        print("recovered",model_save_dir + args.model_restore)
        agent_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                        Policy.acts: expert_actions})
        for iAction in range(5):
            action_prob_list[:,iAction] = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                        Policy.acts: np.ones(len(expert_actions))*iAction})

        if state_only:
            est_reward = np.squeeze(D.get_rewards(expert_observations))
       
        ## temporary change ###########################
        elif weighted_y:
            print("by-action rewards")
            y_list = np.zeros((len(expert_actions),5))
            for i in range(5):
                expert_actions = np.ones((len(expert_actions)))*i
                agent_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                                Policy.acts: expert_actions})
                for iAction in range(5):
                    action_prob_list[:,iAction] = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                                Policy.acts: np.ones(len(expert_actions))*iAction})

                est_reward = D.get_rewards(expert_observations, expert_actions, agent_sa_ph)
                # est_reward2 = sess.run(D.rewards, feed_dict={D.agent_s: expert_observations,
                #                                              D.agent_a: expert_actions,
                #                                              D.agent_sa_p: agent_sa_ph})

                y = np.squeeze(est_reward)
                y_list[:,i] = y

            y_list[np.isinf(y_list)] = -9999
            for iState in range(y_list.shape[0]):
                index = y_list[iState,:] == -9999
                y_list[iState,index] = np.max(y_list[iState,:])

            # est_reward = y_list
            weighted_r = y_list * action_prob_list
            print(weighted_r.shape)
            est_reward = np.sum(weighted_r,axis=1)
            ##############################################
        else:
            est_reward = D.get_rewards(expert_observations, expert_actions, agent_sa_ph)

    # print("get_reward_shape", est_reward.shape, agent_sa_ph.shape, action_prob_list.shape)
        
    return est_reward,agent_sa_ph,action_prob_list #,weighted_y
    