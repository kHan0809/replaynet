import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K

import actor
import critic
import buffer
import Replay_network
class DDPG():

    def __init__(self, state_dim, action_dim):

        self.action_dim = action_dim
        self.state_dim = state_dim

        #Control hyperparameters
        self.GAMMA = 0.95
        self.TAU = 0.05
        self.BUFFER_SIZE = int(1e5)
        self.BATCH_SIZE = 100
        self.REPLAY_BATCH_SIZE = 50
        self.pred_num = 10

        self.actor = actor.Actor(self.state_dim, self.action_dim, self.TAU)
        self.critic = critic.Critic(self.state_dim, self.action_dim, self.TAU)
        self.buffer = buffer.Buffer(self.state_dim, self.action_dim, self.BUFFER_SIZE)
        self.replay = Replay_network.replay_net(self.state_dim, self.state_dim, self.action_dim)
        self.replay_net_buffer = buffer.Buffer(self.state_dim, self.action_dim, self.REPLAY_BATCH_SIZE*self.pred_num)


    def train(self,using_replay = False):
        ##Get batch info & policy
        if using_replay == False:
            states, actions, rewards, next_states, dones = self.buffer.get_batch(self.BATCH_SIZE)
        else:
            self.replay_predict()
            states, actions, rewards, next_states, dones = self.replay_net_buffer.get_batch(self.REPLAY_BATCH_SIZE)

        policy_action = self.actor.get_policy(states) ##tensor
        targ_policy_action = self.actor.get_targ_policy(next_states) #array

        ##Actor train
        q_grads = self.critic.get_Q_gradient(states, policy_action)
        self.actor.actor_train(states, q_grads)

        ##Critic train

        targ_Q = self.critic.get_targ_Q(next_states, targ_policy_action)
        target_y = rewards + self.GAMMA * (1 - dones) * targ_Q
        self.critic.critic_train(states, actions, target_y)

    def replay_train(self):
        max_states, max_actions, max_next_states,_,_= self.buffer.get_sub_opt_state(self.REPLAY_BATCH_SIZE)
        self.replay.replay_net.train_on_batch([max_states, max_next_states], [max_actions])
    def replay_predict(self):
        max_states, max_actions, max_next_states,max_rewards,max_dones = self.buffer.get_sub_opt_state(self.REPLAY_BATCH_SIZE)
        max_actions = self.replay.replay_net.predict([max_states, max_next_states])
        max_act_mean = max_actions[:,:self.action_dim]
        max_act_var = max_actions[:,self.action_dim:]

        for i in range(self.REPLAY_BATCH_SIZE):
            for j in range(self.pred_num):
                temp_act = tf.random.normal([self.action_dim],mean=max_act_mean[i],stddev=tf.exp(max_act_var[i]))
                temp_act = temp_act.numpy()

                self.replay_net_buffer.store(max_states[i],temp_act,max_rewards[i],max_next_states[i],max_dones[i])

        
    def targ_update(self):
        self.critic.target_Q_update()
        self.actor.target_policy_update()

    def action_predict(self):
        states, actions, rewards, next_states, dones = self.buffer.get_batch(self.BATCH_SIZE)
        means, log_stds = self.replay.predict(states,next_states)
        tf.random.normal([1], means, tf.math.exp(log_stds), tf.float32)
