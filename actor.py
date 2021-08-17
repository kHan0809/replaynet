import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K

class Actor():

    def __init__(self, state_dim, action_dim, TAU):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.TAU = TAU

        self.policy = self._gen_network()
        self.targ_policy = self._gen_network()
        
        self.targ_policy.set_weights(self.policy.get_weights())

        self.opt = KO.Adam(learning_rate=0.001)

    def _gen_network(self):
        model = K.Sequential([
            K.Input(shape=self.state_dim),
            KL.Dense(300, activation='relu'),
            #KL.BatchNormalization(),
            KL.Dense(400, activation='relu'),
            #KL.BatchNormalization(),
            KL.Dense(self.action_dim, activation='tanh')
        ])
        model.summary()
        return model

    #Interact with Env
    def get_action(self, states): ##array
        # noise = np.random.normal(-0.2,0.2,self.action_dim)
        # states = states.reshape(-1,3)                      ##
        action = self.policy.predict(states)
        return action

    #For Updating target(y) & policy
    def get_policy(self, states):   ##tensor
        return self.policy(states)

    def get_targ_policy(self, states):   ##arrayy
        return self.targ_policy.predict(states)

    def actor_train(self, states, q_grads):
        q_grads = tf.squeeze(q_grads)
        with tf.GradientTape() as tape:
            grads = tape.gradient(self.get_policy(states), self.policy.trainable_variables, -q_grads)
        
        self.opt.apply_gradients(zip(grads, self.policy.trainable_variables))

    def target_policy_update(self):
        theta = np.array(self.policy.get_weights(),dtype=object)
        targ_theta = np.array(self.targ_policy.get_weights(),dtype=object)
        self.targ_policy.set_weights(self.TAU * theta + (1-self.TAU) * targ_theta)


