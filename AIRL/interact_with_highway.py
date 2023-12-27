import numpy as np
from AIRL.sampler_v2 import batch_sampler
from AIRL.utils import StructEnv_AIRL
from AIRL.policy_net_continuous_discrete import Policy_net
# from AIRL.AIRL_net_discriminator_blend import Discriminator
from shared.transform_obs import transform_obs_v2 as transform_obs
import ray
import os

import highway_irl_v2
from behavior_v2.irl_graphics_v2 import EnvViewer2

import gym

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ray.init(log_to_driver=False)
# ray.init()

@ray.remote
def AIRL_test_function_gym(args, network_policy_values, network_discrim_values, discrete_env_check, EPISODE_LENGTH,
                           i, num_batches):
    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()

    sampler = batch_sampler()
    
    try:
        state_only = args.state_only
        print("state_only loaded",state_only)
    except:
        state_only = False
        
    if state_only:
        from AIRL.AIRL_state_only_discriminator import Discriminator
    else:
        from AIRL.AIRL_net_discriminator_blend import Discriminator
        
    if args.envs_base == "Highway":
        import highway_irl_v2
        env = gym.make('IRL-v2')
        env_discrete = gym.make('IRL-v2')
            
        env.config=np.load('env_configure/'+args.config, allow_pickle = True).tolist()
        env_discrete.config=np.load('env_configure/'+args.config, allow_pickle = True).tolist()
        
        try:
            max_speed = env.config['max_speed']
            print("loaded max_speed:",max_speed)
        except:
            max_speed = 60 #default speed
            print("default max_speed:",max_speed)
        
        env.configure({
            "seed":np.random.randint(0,10000),
        })
        env_discrete.configure({
            "seed":np.random.randint(0,10000),
        })
        #env.reset()
        os.environ["SDL_VIDEODRIVER"] = "dummy" #dummy video device
        if env.viewer is None:
            env.viewer = EnvViewer2(env)
            env_discrete.viewer = EnvViewer2(env_discrete)
        
        env = StructEnv_AIRL(env) #gym.make(args_envs))
        env_discrete = StructEnv_AIRL(env_discrete) #only for getting reward from discrete action, not for training
    else:
        env = StructEnv_AIRL(gym.make(args_envs))
        if discrete_env_check:
            env_discrete = StructEnv_AIRL(gym.make(args_envs))
            
    env.reset()
    #env.seed(0)
    lane_speed = [20,20,20] #initial speed
    #print(lane_speed)
    
    if discrete_env_check:
        env_discrete.reset()
        #env_discrete.seed(0)
        lane_speed_d = [20,20,20] #initial speed
        
    n_feature = 11

    
    Policy_a = Policy_net('Policy_a_{}'.format(i), env, args.units_p, args.units_v, n_feature=n_feature)
    Discrim_a = Discriminator('Discriminator_a_{}'.format(i), env, args.lr_discrim, num_batches, n_feature=n_feature, n_units = args.units_d)

    network_policy_param = Policy_a.get_trainable_variables()
    network_discrim_param = Discrim_a.get_trainable_variables()

    network_policy_operation = []
    for i in range(len(network_policy_param)):
        network_policy_operation.append(tf.assign(network_policy_param[i], network_policy_values[i]))

    network_discrim_operation = []
    for i in range(len(network_discrim_param)):
        
        # print("shape comparison",network_discrim_param[i].shape,network_discrim_values[i].shape)
        network_discrim_operation.append(tf.assign(network_discrim_param[i], network_discrim_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(network_policy_operation)
        sess.run(network_discrim_operation)

        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []
        reward_episode_counter_d = []
        reward_episode_counter_airl = []
        done_count = 0

        while True:

            obs_a = env.observation_type.observe()
            state,lane_speed = transform_obs(obs_a,lane_speed,args.discretize,max_speed)
            
            episode_length += 1
            act, v_pred = Policy_a.act(obs=[state])
            #print("ep vs step",episode_length,env.steps)
            if discrete_env_check:
                act = act.item()
                
                obs_discrete = env_discrete.observation_type.observe()
                state_d,lane_speed_d = transform_obs(obs_discrete,lane_speed_d,args.discretize,max_speed)
                
                act_discrete, v_discrete =  Policy_a.act_discrete(obs=[state_d])
                act_discrete = act_discrete.item()
            else:
                act = np.reshape(act, env.action_space.shape)
                #act = np.reshape(act_discrete, env.action_space.shape)

            v_pred = v_pred.item()
            
            # print("act",act)
            next_obs, reward, done, info = env.step(act)
            #print("done",done,env.time_over)
            #print(act,"state",state,"reward",reward,"done",done,"key",env.prevent_key)
            
            #next_state,lane_speed = transform_obs(next_obs,lane_speed)
            
            if discrete_env_check:
                next_obs_d, reward_d, done_d, info_d = env_discrete.step(act_discrete)
                #next_state_d,lane_speed_d = transform_obs(next_obs_d,lane_speed_d)

            agent_sa_ph = sess.run(Policy_a.act_probs, feed_dict={Policy_a.obs: [state.copy()],
                                                                  Policy_a.acts: [act]})
            
            if state_only:
                reward_a = Discrim_a.get_rewards([state.copy()])
            else:
                reward_a = Discrim_a.get_rewards([state.copy()], [act], agent_sa_ph)

            #print(next_state,reward)
            #print("obs",[env.obs_a.copy()],'act', [act], 'prob',agent_sa_ph)
            reward_a = reward_a.item()
            env.step_airl(reward_a)

            sampler.sampler_traj(state.copy(), act, reward_a, v_pred)

            # if render:
            #     env.render()
            
            if done:
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                #print("reward",reward,"env",env.get_episode_reward())
                reward_episode_counter_airl.append(env.get_episode_reward_airl())
                #print("finished step",env.steps,"i_episode",done_count,"i_step",episode_length)
                env.configure({
                    "seed":np.random.randint(0,10000),
                })
                env.reset()
                done_count+=1
            else:
                env.obs_a = next_obs.copy()
            
            if discrete_env_check:
                if done_d:
                    reward_episode_counter_d.append(env_discrete.get_episode_reward())
                    env_discrete.configure({
                        "seed":np.random.randint(0,10000),
                    })
                    env_discrete.reset()
                else:
                    env_discrete.obs_a = next_obs_d.copy()
            else:
                reward_episode_counter_d.append(0) #zero discrete reward if not discrete environment

            if episode_length >= EPISODE_LENGTH:
                next_state,lane_speed = transform_obs(next_obs,lane_speed,args.discretize,max_speed)
                last_value = np.asscalar(Policy_a.get_value([next_state]))
                sampler.sampler_total(last_value)
                env.reset()
                if discrete_env_check:
                    env_discrete.reset()
                break
                
    #print("counter",reward_episode_counter, episode_length,"done",done_count)
    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter), np.mean(reward_episode_counter_airl), np.mean(reward_episode_counter_d)