import tensorflow as tf
import sys # for now?

from nlptools import unk, STOP

# hyper-parameters:
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

""" A lot (most) of this is borrowed from the tensorflow tutorial on LSTM-based langmods """
class ClickbaitLangMod:

  def __init__(self, train_file):
    self._train = train_file

    self._corpus = self._processCorpus()
    self._vocab = self._makeWordIDs() # map word to int id
    self.vocab_size = len(self._vocab.keys())

    # start describing our RNN:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, tf.float32)
    self._inpt = tf.placeholder(tf.int32, [None, num_steps, None])
    self._output = tf.placeholder(tf.int32, [None])

    # word embeddings:
    E = tf.Variable(tf.truncated_normal([self.vocab_size, hidden_size], stddev=0.1))
    Elookup = tf.nn.embedding_lookup(E, self._inpt)

    # Unroll the network and pass data to it:
    state = self._initial_state
    outputs = [] # list of outputs
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

    # softmax layer:
    sm_weights = tf.get_variable("sm_weights", [size, self.vocab_size], dtype=tf.float32)
    sm_biases = tf.get_variable("sm_biases", [vocab_size], dtype=tf.float32)
    logits = tf.matmul(output, sm_weights) + sm_biases

    # loss function:
    loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._output, [-1])],
                            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

  def train(self):
    pass

  def generateClickbait(self, n):
    return []

  def evaluateTitle(self, title):
    return 0.0

  """ Make the mapping of a word to unique integer id """
  def _makeWordIDs(self):
    wordIDs = {}
    index = 0
    for word in self._corpus:
      if word not in wordIDs:
        wordIDs[word] = index
        index += 1

    return wordIDs # maps word -> int

  """ Read in a tokenised text file, unk it and return list of words """
  def _processCorpus(self):
    with open(self._train, "r") as f:
      text = [ STOP ] # start with stop symbol
      for line in f:
        text.extend(line.split())
        text.append(STOP) # every sentence ends in a stop

    counts = {} # map word->count
    for w in text:
      counts[w] = counts.get(w, 0) + 1 # increment count of this word

    unker = unk.BasicUnker(text, counts) # unk anything with count <= 1
    corpus = unker.getUnkedCorpus() # does what it says

    return corpus
