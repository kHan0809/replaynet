import numpy as np

class Buffer():

    def __init__(self,state_dim,action_dim,buffer_size):

        self.states = np.zeros([buffer_size, state_dim], dtype=np.float32)
        self.actions = np.zeros([buffer_size, action_dim], dtype=np.float32)
        self.rewards = np.zeros([buffer_size,1], dtype=np.float32)
        self.next_states = np.zeros([buffer_size, state_dim], dtype=np.float32)
        self.dones = np.zeros([buffer_size,1], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size
        #여기부터 내가 새로 추가한 것들
        self.states_dim = state_dim
        self.action_dim = action_dim
        #state_cost buffer
        self.states_cost = 100*np.ones([buffer_size, 2], dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get_batch(self, batch_size):
        idxs = np.random.randint(0,self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]

    def get_sub_opt_state(self, batch_size):
        ptr = 0
        count = 0
        max_state      = np.zeros([batch_size, self.states_dim])
        max_next_state = np.zeros([batch_size, self.states_dim])
        max_action     = np.zeros([batch_size, self.action_dim])
        max_rewards    = np.zeros([batch_size, 1], dtype=np.float32)
        max_dones      = np.zeros([batch_size,1], dtype=np.float32)
        for i in range(self.size):
            self.states_cost[i, 0] = np.sqrt(np.square(self.states[i,1]-(-3.141592/9))+np.square(self.states[i,2]-(1.05/2))+np.square(self.states[i,3]-(0.785/5))\
                                             +np.square(self.states[i,4]-(-0.4))+np.square(self.states[i,5]-(0.7/2))+np.square(self.states[i,6]-(0))\
                                             +np.square(self.states[i,7]-(0))+np.square(self.states[i,8]-(5)))
            self.states_cost[i, 1]  = np.sqrt(np.square(self.next_states[i,1]-(-3.141592/9))+np.square(self.next_states[i,2]-(1.05/2))+np.square(self.next_states[i,3]-(0.785/5))\
                                             +np.square(self.next_states[i,4]-(-0.4))+np.square(self.next_states[i,5]-(0.7/2))+np.square(self.next_states[i,6]-(0))\
                                             +np.square(self.next_states[i,7]-(0))+np.square(self.next_states[i,8]-(5)))

        for i in self.states_cost[:,1].argsort():
            if self.states_cost[i,0]>self.states_cost[i,1]:
                # print("===============")
                # print(i)
                # print("===================")
                # print(self.states_cost[i,0])
                # print(self.states_cost[i, 1])
                # print(self.states[i])
                # print(self.next_states[i])
                max_state[ptr]      = self.states[i]
                max_next_state[ptr] = self.next_states[i]
                max_action[ptr]     = self.actions[i]
                max_rewards[ptr]    = self.rewards[i]
                max_dones[ptr]      = self.dones[i]
                ptr = ptr + 1
                count = count + 1
            if count == batch_size:
                break

        # for i in self.rewards.argsort()[-batch_size:]:
        #
        #     print(self.next_states[i])
        #     print("asdfasdfsadfasdfasdfsadfsf")
        #     max_next_state.append(self.next_states[i])
        return max_state, max_action, max_next_state, max_rewards, max_dones
