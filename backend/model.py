import tensorflow as tf
import sys # for now?
import codecs

from nlptools import unk, STOP

# hyper-parameters:
batch_size = 20
hidden_size = 200
embed_size = 30
num_layers = 3
max_length = 20

""" A lot (most) of this is borrowed from the tensorflow tutorial on LSTM-based langmods """
class ClickbaitLangMod:

  def __init__(self, train_file):
    self._train = train_file

    text_corpus = self._processCorpus()
    self._vocab = self._makeWordIDs(text_corpus) # map word to int id
    self._corpus = [[self._vocab[w] for w in line] for line in text_corpus]
    self.vocab_size = len(self._vocab.keys())
    print(self.vocab_size)

    # inputs and outputs:
    self._inpt = tf.placeholder(tf.int32, [1, max_length])
    self._targets = tf.placeholder(tf.int32, [None])

    # word embeddings:
    E = tf.Variable(tf.truncated_normal([self.vocab_size, embed_size], stddev=0.1))
    Elookup = tf.nn.embedding_lookup(E, self._inpt)

    # start describing our RNN:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    # simulate time steps:
    output, state = tf.nn.dynamic_rnn(cell, Elookup, dtype=tf.float32)
    output = tf.reshape(output, [-1, hidden_size])

    # softmax layer:
    sm_weights = tf.Variable(tf.truncated_normal([hidden_size, self.vocab_size], stddev=0.1))
    sm_biases = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]))
    self._logits = tf.matmul(output, sm_weights) + sm_biases
    self._cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self._logits, self._targets)

    # define the training step:
    self._train_step = tf.train.AdamOptimizer(1e-4).minimize(self._cross_entropy)
    perplexity = tf.exp(tf.reduce_mean(self._cross_entropy))

    # setup the tf session:
    self.sess = tf.Session()
    self.sess.run(tf.initialize_all_variables())

  def train(self):
    #total = int(len(self._corpus)/batch_size)
    total = len(self._corpus)
    n = 0
    for l in self._corpus:
      words = l[:-1]
      nextwords = l[1:]
      if len(words) <= max_length:
        words += [0]*(max_length - len(words))
        nextwords += [0]*(max_length - len(nextwords))

        self._train_step.run(feed_dict={self._inpt: [words], self._targets: nextwords}, session=self.sess)
        #if not n%100:
        print("Batch #%d of %d (%.2f%%):"%(n,total,100*n/total))
        #print("\tTrain perplexity: %.6f"%perplexity.eval(feed_dict={inpt:words, output:nextwords}, session=sess))

      n+=1

  def generateClickbait(self, n):
    stopcode = self._vocab[STOP]
    sentence = [ stopcode ]
    maxlen = int(np.random.randn()*2 + MAX_LENGTH)
    while len(sentence) < maxlen:
      word = sentence[-1]
      dist = np.array(probs.eval(feed_dict={self._inpt: [word]}, session=sess)[0])
      dist /= dist.sum()
      nword = np.random.choice(len(dist),p=dist)

      if nword == stopcode:
        break
      sentence.append(nword)

    return sentence[1:] # exclude stop symbol

  def evaluateTitle(self, title):
    return 0.0

  """ Make the mapping of a word to unique integer id """
  def _makeWordIDs(self, text):
    wordIDs = {}
    index = 1
    for line in text:
      for word in line:
        if word not in wordIDs:
          wordIDs[word] = index
          index += 1

    return wordIDs # maps word -> int

  """ Read in a tokenised text file, unk it and return list of words """
  def _processCorpus(self):
    with codecs.open(self._train, "r",encoding='utf-8', errors='ignore') as f:
      text = [(STOP+line+STOP).split() for line in f]

    counts = {} # map word->count
    for l in text:
      for w in l:
        counts[w] = counts.get(w, 0) + 1 # increment count of this word

    unker = unk.BasicUnker(text, counts) # unk anything with count <= 1
    corpus = unker.getUnkedCorpus() # does what it says

    return corpus

