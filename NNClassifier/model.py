#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
import datetime
import os
from . import config
#from tensorflow.models.rnn_* import rnn
#import numpy as np

class RNNClassifierArgs(object):
    def __init__(self,batch_size = 4, num_steps = 128, num_layers = 2, learning_rate = 0.01, max_grad_norm = 5.,
                    init_scale = 0.05, hidden_size = 128, keep_prob = .5, word_embedding_size = 32,
                    word_vocab_size = 128, num_drug_classes = 33,
                    log_dir=os.path.join(config.base_dir,"tmp/DrugLog"+datetime.datetime.now().isoformat()),
                    lambda_loss_amount = 0.005):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.init_scale = init_scale
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.word_embedding_size = word_embedding_size
        self.word_vocab_size = word_vocab_size
        self.num_drug_classes = num_drug_classes
        self.log_dir=log_dir
        self.lambda_loss_amount = lambda_loss_amount
    
    def __repr__(self):
        return (str(self.__dict__))

#We define our own fast GRUCell, with softsign gates
class FastGRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.nn.softsign):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, tf.nn.rnn_cell._linear([inputs, state],
                                             2 * self._num_units, True, 1.0))
        # We use (a shifted & rescaled) softsign rather than a sigmoid
        r, u = 0.5*(1+tf.nn.softsign(r)), 0.5*(1+tf.nn.softsign(u))
      with vs.variable_scope("Candidate"):
        c = self._activation(tf.nn.rnn_cell._linear([inputs, r * state],
                                     self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class RNNClassifierModel(object):
    def __init__(self,args, is_training=False, verbose=False):
        # First we store the configuration object as a component.
        self._args = args
        self.keep_prob = tf.constant(self.args.keep_prob)
        
        # Create some placeholders for the input/output data
        self._input_IDs = tf.placeholder(tf.int32, [self.args.batch_size, self.args.num_steps],name="input_IDs") # batch_size x num_steps
        self._target_probs = tf.placeholder(tf.float32, [self.args.batch_size, self.args.num_drug_classes],name="target_probs") # batch_size x classes
                
        if is_training:
            name_scope = "rnnTrainer"
        else:
            name_scope = "rnnClassifier"
        with tf.name_scope(name_scope):
            
            # We now create learnable embeddings for the words
            with tf.name_scope('embedding'):
                # for now, embedding needs to be placed on the cpu.
                with tf.device("/cpu:0"):
                    word_embedding = tf.get_variable("word_embedding",
                                        [self.args.word_vocab_size, self.args.hidden_size],
                                        initializer=tf.random_normal_initializer(0, 1/self.args.hidden_size))
                    inputs = tf.nn.embedding_lookup(word_embedding, self.input_IDs,name="word_vect")
                if is_training and self.args.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, self.keep_prob)
            
            
            # Create the GRU Cell
            size=self.args.hidden_size
            cell=FastGRUCell(size,activation=tf.nn.softsign)
            
            if is_training and self.args.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=self.keep_prob)
            # Now stack the GRU Cell on top of itself self.args.num_layers times
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.args.num_layers)
            
            
            # self._initial_state = tf.tile(tf.concat(1,[tf.zeros([self.args.batch_size, self.args.hidden_size], tf.float32),theme]),
            #                                [1,self.args.num_layers],name="initial_state") 
            #self._initial_state = tf.zeros([self.args.batch_size, self.args.hidden_size], tf.float32,name="initial_state") 
            self._initial_state = cell.zero_state(self.args.batch_size,tf.float32)

            # Link the GRU cell sequentially to itself:
           #  outputs = []
#                 state=self.initial_state
#                 # Make sure we save the scope as rnn_scope so that we can reuse the trained GRU weights later when sampling.
#                 with tf.variable_scope("RNN"):
#                     for time_step in range(self.args.num_steps):
#                         # After the first unit, we want to reuse the GRU weights
#                         if time_step > 0:
#                             tf.get_variable_scope().reuse_variables()
#                         (cell_output, state) = cell(inputs[:, time_step, :], state)
#                         outputs.append(cell_output)
                    
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, self.args.num_steps, inputs)]
            outputs, state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)
            

            self._final_state=state #Save the final state so we can access it later when training.
            
            
            # Transform the GRU output to logits via a learnt linear transform
            output = outputs[-1]#tf.reshape(tf.concat(1, outputs), [-1, size])
            softmax_w = tf.get_variable("softmax_w", [size, self.args.num_drug_classes],initializer=tf.random_normal_initializer(0,1/size))
            softmax_b = tf.get_variable("softmax_b", [self.args.num_drug_classes],initializer=tf.constant_initializer(0.))
            logits = tf.matmul(output, softmax_w) + softmax_b
            self._output_prob=tf.nn.softmax(logits)
            
            
            # Compute the loss
            with tf.name_scope('loss'):
                #weight_list = [theme_strength_reshaped for _ in range(self.args.num_steps)]
                #weights=tf.reshape(tf.concat(1, weight_list), [-1])
                #loss = tf.nn.seq2seq.sequence_loss_by_example(
                #    [logits],
                #    [tf.reshape(self.target_IDs, [-1])],
                #    [tf.ones([self.args.batch_size * self.args.num_steps])])
                #self._cost = tf.reduce_sum(loss) / self.args.batch_size
                self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self._target_probs, logits = logits)
                self._cost = tf.reduce_mean(self._cross_entropy)
                
                self._l1 = self.args.lambda_loss_amount * sum(tf.reduce_sum(tf.abs(tf_var))
                                                            for tf_var in tf.trainable_variables()
                                                            if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
                                                    )
            
            with tf.name_scope('KL_Divergence'):
                with tf.name_scope('correct_prediction'):
                    masked_out = tf.boolean_mask(self._output_prob,tf.logical_not(tf.equal(self._output_prob,tf.constant(0.))))
                    avg_entropy = -tf.reduce_sum(tf.mul(masked_out,tf.log(masked_out)))/size
                    #kl_div = entropy+tf.reduce_mean(self._cross_entropy)
                with tf.name_scope('average_KL_Divergence'):
                    self._average_KL_Divergence = avg_entropy+tf.reduce_mean(self._cross_entropy)
                tf.scalar_summary('average_KL_Divergence', self._average_KL_Divergence)
            if verbose:
                tf.histogram_summary("output weights", softmax_w)
                tf.histogram_summary("output biases", softmax_b)
                tf.histogram_summary("word embedding", word_embedding)
                tf.scalar_summary("output weights zero fraction", tf.nn.zero_fraction(softmax_w))
                tf.scalar_summary("output biases zero fraction", tf.nn.zero_fraction(softmax_b))
                tf.scalar_summary("word embedding zero fraction", tf.nn.zero_fraction(word_embedding))
                tf.scalar_summary('cross entropy', self.cost)
                tf.scalar_summary('l1 loss', self.l1)
                tf.scalar_summary('total loss', self.l1+self.cost)
                # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
                self._merged = tf.merge_all_summaries()
                self._writer = tf.train.SummaryWriter(self.args.log_dir,
                                                        tf.get_default_graph())

            if not is_training:
                return
                
            with tf.name_scope('train'):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost+self.l1, tvars),self.args.max_grad_norm)
                optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
                self._train_op = optimizer.apply_gradients(zip(grads, tvars))
                    
                
                    
    @property
    def args(self):
        return self._args
    
    @property
    def input_IDs(self):
        return self._input_IDs
    
    @property
    def target_probs(self):
        return self._target_probs
        
    @property
    def average_KL_Divergence(self):
        return self._average_KL_Divergence
    
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def cost(self):
        return self._cost
        
    @property
    def l1(self):
        return self._l1
    
    @property
    def final_state(self):
        return self._final_state
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def output_prob(self):
        return self._output_prob
    
    @property
    def merged(self):
        return self._merged
    
    @property
    def writer(self):
        return self._writer
