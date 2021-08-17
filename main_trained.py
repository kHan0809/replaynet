import gym
import matplotlib.pyplot as plt
import numpy as np
import ddpg
import sys
import tensorflow.keras as K
import tensorflow as tf
sys.path.append('C:/Users/owner/.mujoco/mujoco200/bin')

if __name__ == "__main__":

    env_name = "Pendulum-v0" #"InvertedPendulum-v2"


    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_lim = env.action_space.high[0]

    agent = ddpg.DDPG(state_dim, action_dim)

    # agent.actor.policy = tf.keras.models.load_model('./hg/RL/policy_v1.h5')
    agent.actor.policy.save('./hg/RL/test.h5')

    # state, reward, done, ep_rew, ep_len, ep_cnt =
    # env.reset(), 0, False, [0.0], 0, 0
    #
    # for t in range(1000):
    #
    #     env.render()
    #
    #
    #     action = agent.actor.get_action(state.reshape([1, state_dim]))
    #     noise = 0.2 * np.random.randn(action_dim)
    #     action = action_lim * action[0] + noise
    #
    #
    #     next_state, reward, done, _ = env.step(action)
    #     ep_rew[-1] += reward
    #     ep_len += 1
    #
    #
    #
    #     agent.buffer.store(state, action, reward, next_state, done)
    #
    #     state = next_state











