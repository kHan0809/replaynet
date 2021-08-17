import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.optimizers as KO
import tensorflow.keras as K

class replay_net():

    def __init__(self, state_dim, next_state_dim,action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.next_state_dim = state_dim

        self.replay_net = self._gen_network()

    def custom_loss(self):
        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            prediction, log_variance = tf.split(y_pred, [self.action_dim, self.action_dim], 1)
            return tf.reduce_sum(
                0.5 * tf.exp(-1 * log_variance) * tf.square(tf.abs(y_true - prediction)) + 0.5 * log_variance)

        return loss

    def _gen_network(self, units=(400, 200, 200, 100)):
        inputs = [KL.Input(shape=self.state_dim), KL.Input(shape=self.state_dim)]
        concat = KL.Concatenate(axis=-1)(inputs)
        x = KL.Dense(units[0], name="Hidden0", activation="relu")(concat)
        for index in range(1, len(units)):
            x = KL.Dense(units[index], name="L{}".format(index), activation="relu")(x)

        actions_mnv = KL.Dense(self.action_dim*2, name="mean")(x)
        # actions_log_std = KL.Dense(self.action_dim, name="log_std")(x)

        model = K.models.Model(inputs=inputs, outputs=[actions_mnv])
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=self.custom_loss())
        model.summary()
        return  model




    