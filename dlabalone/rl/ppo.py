##########################################################
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Code from: https://keras.io/examples/rl/ppo_cartpole/
# About entropy loss: https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26


policy_learning_rate = 3e-4
EPSILON = 1e-6
ENTROPY_LOSS_FACTOR = 0.001
_policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)


# def logprobabilities_from_logits(logits, a, num_actions):
#     # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
#     logprobabilities_all = tf.nn.log_softmax(logits)
#     logprobability = tf.reduce_sum(
#         tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
#     )
#     return logprobability


def logprobabilities_from_prob(probability, a, num_actions):
    logprobabilities_all = tf.math.log(probability)
    # logprobabilities_all = tf.nn.log_softmax(probability)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


def clip_to_epsilon(x):
    return tf.clip_by_value(x, EPSILON, 1 - EPSILON)


def calculate_pi(probability, action, num_actions):
    return tf.reduce_sum(
            tf.one_hot(action, num_actions) * probability, axis=1
        )


@tf.function
def train_policy(actor, observation_buffer, action_buffer, pi_old_buffer, advantage_buffer, num_actions,
                 clip_ratio=0.2):
    advantage_buffer = tf.where(advantage_buffer > 0, advantage_buffer, advantage_buffer / (num_actions - 1))

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        # Calculate entropy
        # clipped_probabilities = tf.clip_by_value(
        #     actor(observation_buffer),
        #     EPSILON,
        #     1 - EPSILON
        # )
        clipped_probabilities = clip_to_epsilon(actor(observation_buffer))
        entropy = -tf.reduce_mean(tf.math.multiply(clipped_probabilities, tf.math.log(clipped_probabilities)))

        # Calculate loss_clip
        pi = calculate_pi(clipped_probabilities, action_buffer, num_actions)
        ratio = (pi / pi_old_buffer)
        clip = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio), advantage_buffer)
        loss_clip = tf.reduce_sum(tf.minimum(ratio * advantage_buffer, clip))

        # Calculate final loss
        loss = loss_clip + ENTROPY_LOSS_FACTOR * entropy
        loss = tf.negative(loss)

    policy_grads = tape.gradient(loss, actor.trainable_variables)
    _policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    # kl = tf.reduce_mean(
    #     logprobability_buffer
    #     - logprobabilities_from_prob(actor(observation_buffer), action_buffer, num_actions)
    # )
    # kl = tf.reduce_sum(kl)
    # return kl

