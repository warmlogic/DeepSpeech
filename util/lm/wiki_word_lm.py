from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from os import path
from util.lm import wiki_reader

# Based off Zaremba, Sutskever, Vinyals "Recurrent Neural Network Regularization"
# https://arxiv.org/abs/1409.2329v5

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model_path", "./models/lm", "model_path")
flags.DEFINE_string("data_path", "./data/wikipedia", "data_path")
flags.DEFINE_string("model", "large", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("use_checkpoint", False, "A boolean indicating whether to use the checkpointed model")

FLAGS = flags.FLAGS

class WIKIModel(object):
  """The WIKI model."""

  def __init__(self, is_training, config):
    # Init variables
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    dtype = tf.float32

    # Define placeholders
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Define multi-layer RNN operator graph
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    # Init more variables
    self._initial_state = cell.zero_state(batch_size, dtype)

    # Embed self._input_data  in to lower dimension space
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=dtype)
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    # Add dropout if training
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Create RNN from cell
    inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

    # Reshape outputs for final softmax layer
    output = tf.reshape(tf.concat(1, outputs), [-1, size])

    # Create/get softmax variables
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=dtype)

    # Compute softmax of output
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Compute loss from logits
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=dtype)])

    # Average loss across batch
    self._cost = cost = tf.reduce_sum(loss) / batch_size

    # Retain final state
    self._final_state = state

    # If we're not training don't consider gradient computations
    if not is_training:
      return

    # Define learning rate variable
    self._lr = tf.Variable(0.0, trainable=False)

    # Obtain all trainable variables
    tvars = tf.trainable_variables()

    # Compute clipped gradients of tvars
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)

    # Create oprimizer with learning rate self._lr
    optimizer = tf.train.GradientDescentOptimizer(self._lr)

    # Apply gradients to tvars
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Define learning rate placeholder
    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")

    # Assign learning rate to learning rate placeholder
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def run_epoch(session, model, data, eval_op, verbose=False):
  """Runs the model on the given data."""

  # Init variables
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0

  # Get initial state
  state = session.run(model.initial_state)

  # Train the model on the passed data set
  for step, (x, y) in enumerate(wiki_reader.wiki_iterator(data, model.batch_size, model.num_steps)):
    # Define operators to run
    fetches = [model.cost, model.final_state, eval_op]
    # Setup feed_dict params to pass to run
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    # Run operators
    cost, state, _ = session.run(fetches, feed_dict)
    # Track costs and iterations
    costs += cost
    iters += model.num_steps

    # Log progress info
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))

  # Return perplexity
  return np.exp(costs / iters)

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def get_configs():
  # Obtain config info
  config = get_config()

  # Obtain evaluation info
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  return config, eval_config

def _get_models(config, eval_config):
  # Define proper initializer for the model
  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  # Scope trained model variables
  with tf.variable_scope("model", reuse=None, initializer=initializer):
    m = WIKIModel(is_training=True, config=config)
  # Scope validation and test model variables
  with tf.variable_scope("model", reuse=True, initializer=initializer):
    mvalid = WIKIModel(is_training=False, config=config)
    mtest = WIKIModel(is_training=False, config=eval_config)

  return m, mvalid, mtest

def main(_):
  # Check required paramers
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to WIKI data directory")

  # Obtain train, valid, and test data sets
  train_data, valid_data, test_data = wiki_reader.wiki_raw_data(FLAGS.data_path, FLAGS.model_path, FLAGS.use_checkpoint)

  # Obtain config info
  config, eval_config = get_configs()

  # Define graph and session scope for the model
  with tf.Graph().as_default(), tf.Session() as session:
    # Get models
    m, mvalid, mtest = _get_models(config, eval_config)

    # Init variables
    tf.initialize_all_variables().run()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Branch on checkpoint use
    if not FLAGS.use_checkpoint:
      # Train the model for config.max_max_epoch epochs
      for i in range(config.max_max_epoch):
        # Compute and update learning rate
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        # Log learning rate and train & valid perplexity
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, train_data, m.train_op, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      # Log test perplexity
      test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
      print("Test Perplexity: %.3f" % test_perplexity)

      # Save the variables to disk.
      save_path = saver.save(session, path.join(FLAGS.model_path,"model.ckpt"))
      print("Model saved in file: %s" % save_path)
    else:
      ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
      else:
        raise ValueError("Must train and checkpoint a model before loading a checkpoint")

      # Log test perplexity
      test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
      print("Test Perplexity: %.3f" % test_perplexity)

def get_language_models():
  # Obtain config info
  config, eval_config = get_configs()

  # Return models
  return _get_models(config, eval_config)

def load_models(session, saver):
  # Load models from disk
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    raise ValueError("Must train and checkpoint a model before loading a checkpoint")

def get_perplexities(session, mtest, texts):
  # Define perplexities to return
  perplexities = []

  # Get word_to_id dictionary
  word_to_id = wiki_reader.get_word_to_id(FLAGS.model_path)

  # Loop over texts
  for text in texts:
    # Convert text to data set with word id's
    test_data = wiki_reader.text_to_word_ids(text, word_to_id)

    # Compute perplexity
    perplexities.append(run_epoch(session, mtest, test_data, tf.no_op()))

  return perplexities

if __name__ == "__main__":
  tf.app.run()
