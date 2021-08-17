import gym
import matplotlib.pyplot as plt
import numpy as np
import ddpg
import sys
sys.path.append('C:/Users/owner/.mujoco/mujoco200/bin')

if __name__ == "__main__":
    Algo_list = ['orig','replay']
    reward_buff_orig = []
    reward_buff_my = []
    for Algo in Algo_list:
        for j in range(2):
            env_name = "HalfCheetah-v2" #"InvertedPendulum-v2" "HalfCheetah-v2"

            MAX_EPISODE_LENGTH = 1000
            MAX_SETPS_INTERACTION = 1000000
            START_STEPS = 10000


            env = gym.make(env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            action_lim = env.action_space.high[0]

            agent = ddpg.DDPG(state_dim, action_dim)

            state, reward, done, ep_rew, ep_len, ep_cnt = env.reset(), 0, False, [0.0], 0, 0


            for t in range(MAX_SETPS_INTERACTION):

                env.render()

                if t > START_STEPS:
                    action = agent.actor.get_action(state.reshape([1, state_dim]))
                    noise = 0.2 * np.random.randn(action_dim)
                    action = action_lim * action[0] + noise
                else:
                    action = env.action_space.sample()
                # print(state)

                next_state, reward, done, _ = env.step(action)
                ep_rew[-1] += reward
                ep_len += 1

                done = False

                if ep_len == MAX_EPISODE_LENGTH:
                    done = True

                agent.buffer.store(state, action, reward, next_state, done)

                state = next_state

                if done:
                    ep_cnt += 1
                    print(f"Episode: {ep_cnt}, Reward: {ep_rew[-1]}")
                    ep_rew.append(0.0)

                    if t > 5000:
                        if (Algo == 'replay') & (t < 50000):
                            for i in range(80):
                                agent.replay_train()
                        for i in range(200):
                            agent.train(False)

                            if (i % 5) == 4 :
                                agent.targ_update()
                    if Algo == 'replay':
                        if (t >= 8000) & (t < 200000):
                            for i in range(50):
                                agent.train(True)

                    print("upgrade complete {} times".format(ep_cnt))

                    state, reward, done, ep_len = env.reset(), 0, False, 0
            env.close()
            # agent.actor.policy.save('./hg/RL/policy_v1.h5')
            # agent.critic.Q.save('./hg/RL/critic_v1.h5')

            if Algo == 'orig':
                reward_buff_orig.append(ep_rew[:-1])
            else:
                reward_buff_my.append(ep_rew[:-1])


    x = np.arange(1, len(ep_rew[:-1])+1)

    orig_m = np.mean(reward_buff_orig, 0)
    orig_v = np.var(reward_buff_orig,0)/10


    my_m = np.mean(reward_buff_my, 0)
    my_v = np.var(reward_buff_my,0)/10
    #Plot learning curve
    plt.style.use('seaborn')
    plt.plot(x,orig_m,label = 'Orig')
    plt.plot(x, my_m, label='My')
    plt.fill_between(x, orig_m - orig_v, orig_m + orig_v, alpha=0.1)
    plt.fill_between(x, my_m - my_v, my_m + my_v, alpha=0.1)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()







