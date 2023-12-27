#import tensorflow as tf
import gym
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Discriminator:
    def __init__(self, name: str,  env, lr, num_batches,n_feature = 8,n_units = 64, optim = 'Adam', swag = False):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        if swag:
            self.lr = tf.placeholder_with_default(lr, shape=[], name=None) #only if swag?
        else:
            self.lr = lr
            print("fixed lr_discrim",lr)
        
        if env.unwrapped.spec.id == 'IRL-v1' or env.unwrapped.spec.id == 'IRL-v2':
            ob_space = np.zeros(n_feature)
        else:
            ob_space = env.observation_space
        
        self.obs_t = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape))
        self.nobs_t = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape))
        
        self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
        self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
        
        self.gamma = 1.0

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name

            with tf.variable_scope('reward'):
                self.reward = self.reward_network(self.obs_t,n_units = n_units)
            
            with tf.variable_scope('value') as value_scope:
                h_ns = self.value_network(self.nobs_t,n_units = n_units)
                value_scope.reuse_variables()
                self.h_s = h_s = self.value_network(self.obs_t,n_units = n_units)
                
            log_f = self.reward + self.gamma*h_ns - h_s
            self.f_reward = tf.exp(log_f) #f_reward
            log_p = self.lprobs

            log_fp = tf.reduce_logsumexp([log_f, log_p], axis=0)
            self.discrim_output = tf.exp(log_f-log_fp)
            
            with tf.variable_scope('loss'):
                self.loss = loss = -tf.reduce_mean(self.labels*(log_f-log_fp) + (1-self.labels)*(log_p-log_fp))
            
            d_net_trainable = self.get_trainable_variables()
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(loss, var_list=d_net_trainable)

    def reward_network(self, input, n_units):
        layer_1 = tf.layers.dense(inputs=input, units=n_units, activation=tf.nn.leaky_relu, name='g_layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=n_units, activation=tf.nn.leaky_relu, name='g_layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=n_units, activation=tf.nn.leaky_relu, name='g_layer3')
        out = tf.layers.dense(inputs=layer_3, units=1, name='g_out')
        return out
    
    def value_network(self, input, n_units):
        layer_1 = tf.layers.dense(inputs=input, units=n_units, activation=tf.nn.leaky_relu, name='h_layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=n_units, activation=tf.nn.leaky_relu, name='h_layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=n_units, activation=tf.nn.leaky_relu, name='h_layer3')
        out = tf.layers.dense(inputs=layer_3, units=1, name='h_out')
        return out
    
    def train_state(self, obs_t, nobs_t, lprobs, labels):
        return tf.get_default_session().run(self.train_op, feed_dict={self.obs_t: obs_t,
                                                                      self.nobs_t: nobs_t,
                                                                      self.lprobs: lprobs,
                                                                      self.labels: labels})
    
    # def get_rewards(self, agent_s, agent_a, agent_sa_p):
    #     return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
    #                                                                  self.agent_a: agent_a,
    #                                                                  self.agent_sa_p: agent_sa_p})
    
    # g reward
    def get_rewards(self, obs_t):
        scores = tf.get_default_session().run(self.reward, feed_dict={self.obs_t: obs_t})
        return scores
    
    # f reward
    def get_f_rewards(self):
        scores = tf.get_default_session().run(self.f_reward, feed_dict={self.obs_t: obs_t,
                                                                            self.nobs_t: kwargs['nobs_t']
                                                                            })
        return scores

    # log reward:logD-log(1-D)
    def get_l_rewards(self, obs_t, nobs_t, lprobs):
        scores = tf.get_default_session().run(self.discrim_output, feed_dict={self.obs_t: obs_t,
                                                                            self.nobs_t: nobs_t,
                                                                            self.lprobs: lprobs
                                                                            })
        scores = np.log(scores) - np.log(1-scores)
        return scores

    def get_trainable_variables(self): #this might be used to design the regulator
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)