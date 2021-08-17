import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K

class Critic():
    def __init__(self, state_dim, action_dim, TAU):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.TAU = TAU

        self.Q = self._gen_network()
        self.targ_Q = self._gen_network()

        self.targ_Q.set_weights(self.Q.get_weights())

    def _gen_network(self):

        # state_input = KL.Input(shape=self.state_dim)
        # action_input = KL.Input(shape=(self.action_dim,))

        state_input = KL.Input((self.state_dim,))
        s1 = KL.Dense(300, activation='relu')(state_input)
        #s1_bn = KL.BatchNormalization()(s1)
        s2 = KL.Dense(400, activation='relu')(s1)
        #s2_bn = KL.BatchNormalization()(s2)

        action_input = KL.Input((self.action_dim,))
        a1 = KL.Dense(300, activation='relu')(action_input)
        #a1_bn = KL.BatchNormalization()(a1)


        dense1 = KL.concatenate([s2, a1], axis=-1)
        #dense1_bn = KL.BatchNormalization()(dense1)
        dense2 = KL.Dense(300, activation='relu')(dense1)
        #dense2_bn = KL.BatchNormalization()(dense2)
        output = KL.Dense(1, activation='linear')(dense2)

        model = K.Model(inputs = [state_input, action_input], outputs = output)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss = 'mse')

        model.summary()
        return model
    
    #For updating policy, actions follow policy & For updating Q-function, actions follow replay buffer.
    def get_Q(self, states, actions):         ##tensor
        return self.Q([states, actions])

    #For computing targets, use next_state & target action for next_state by target policy
    def get_targ_Q(self, states, actions):    ##array
        #states = tf.convert_to_tensor(states)
        return self.targ_Q.predict([states, actions])

    def get_Q_gradient(self, states, policy_action):
        states = tf.convert_to_tensor(states)
        with tf.GradientTape() as tape:
            tape.watch(policy_action)
            Q = self.get_Q(states, policy_action)   
        return tape.gradient(Q, policy_action)

    def critic_train(self, states, actions, target_y):
        self.Q.train_on_batch([states, actions], [target_y])

    def target_Q_update(self):
        pi = np.array(self.Q.get_weights(),dtype=object)
        targ_pi = np.array(self.targ_Q.get_weights(),dtype=object)
        self.targ_Q.set_weights(self.TAU * pi + (1-self.TAU) * targ_pi)