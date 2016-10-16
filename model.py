import tensorflow as tf
import sys # for now?
import codecs
import pickle
import numpy as np
import nltk

from nlptools import unk, STOP

# hyper-parameters:
batch_size = 20
hidden_size = 200
embed_size = 30
num_layers = 3
max_length = 20

""" This one doesn't use an LSTM but loads a pre-trained tensorflow graph
     based on a non-linear bigram language model
"""
class ShittyClickbaitLangMod:
  def __init__(self, model_file):
    self._sess = tf.Session()
    with open("%s.dict"%model_file, "rb") as f:
      self._vocab = pickle.load(f)
      
    self.vocab_size = len(self._vocab.keys())
    self._inv_map = {v: k for k, v in self._vocab.items()} # inverse map int id->string

    saver = tf.train.import_meta_graph("%s.meta"%model_file)
    saver.restore(self._sess, model_file) # restore the session

    self._inpt = tf.get_collection('inpt')[0]
    self._output = tf.get_collection('output')[0]
    self._logits = tf.get_collection('logits')[0]
    self._perplexity = tf.get_collection('perplexity')[0]

    noise = tf.random_normal(tf.shape(self._logits),stddev=0.0)
    self._probs = tf.nn.softmax(self._logits + noise) # convert to probabilities

  def train(self):
    pass # already trained

  def generateClickbait(self, n):
    stopcode = self._vocab[STOP]
    #sentence = [ np.random.randint(vocab_size) ]
    sentence = [ stopcode ]
    while len(sentence) < n:
      word = sentence[-1]
      dist = np.array(self._probs.eval(feed_dict={self._inpt: [word]}, session=self._sess)[0])
      dist /= dist.sum()
      nword = np.random.choice(len(dist),p=dist)

      if nword == stopcode:
        break
      sentence.append(nword)

    s = sentence[1:] # exclude stop symbol
    return "".join([self._inv_map[w]+" " for w in s]).strip()

  """ Title as string """
  def evaluateTitle(self, title):
    title = nltk.word_tokenize("%s %s %s"%(STOP,title,STOP))
    words = [self._vocab.get(w, 0) for w in title[:-1]]
    nextwords = [self._vocab.get(w, 0) for w in title[1:]]
    return self._perplexity.eval(feed_dict={self._inpt:words, self._output:nextwords}, session=self._sess)

""" A lot (most) of this is borrowed from the tensorflow tutorial on LSTM-based langmods """
class ClickbaitLangMod:

  """ train_file: path to text corpus, or path to saved model.
      saved: if saved is False (default), train_file is a path to corpus from which to train
              otherwise try to load the graph from that path
  """
  def __init__(self, train_file, saved=False):
    self._sess = tf.Session()
    self._training = False
    if saved:
      with open("%s.dict"%train_file, "rb") as f:
        self._vocab = pickle.load(f)
        
      self.vocab_size = len(self._vocab.keys())

      saver = tf.train.import_meta_graph("%s.meta"%train_file)
      saver.restore(self._sess, train_file) # restore the session

      self._inpt = tf.get_collection('input')[0]
      self._output = tf.get_collection('targets')[0]
      self._logits = tf.get_collection('logits')[0]
      self._cross_entropy = tf.get_collection('cross_entropy')[0]
    else:
      self._training = True
      self._train = train_file

      text_corpus = self._processCorpus()
      self._vocab = self._makeWordIDs(text_corpus) # map word to int id
      self._corpus = [[self._vocab[w] for w in line] for line in text_corpus]
      self.vocab_size = len(self._vocab.keys()) + 1 # include invalid word

      self._inv_map = {v: k for k, v in self._vocab.items()} # inverse map int id->string
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

      # setup the tf session:
      self.sess.run(tf.initialize_all_variables())

    # for inference and analysis:
    self._perplexity = tf.exp(tf.reduce_mean(self._cross_entropy))
    noise = tf.random_normal(tf.shape(self._logits),stddev=1.0)
    self._probs = tf.nn.softmax(self._logits + noise) # convert to probabilities

    self._inv_map = {v: k for k, v in self._vocab.items()} # inverse map int id->string

  def train(self):
    if self._training:
      #total = int(len(self._corpus)/batch_size)
      total = len(self._corpus)
      n = 0
      for l in self._corpus:
        words = l[:-1]
        nextwords = l[1:]
        if len(words) <= max_length:
          words += [0]*(max_length - len(words)) # 0 is "invalid", doesn't map to real word
          nextwords += [0]*(max_length - len(nextwords))

          self._train_step.run(feed_dict={self._inpt: [words], self._targets: nextwords}, session=self.sess)
          #if not n%100:
          print("Batch #%d of %d (%.2f%%):"%(n,total,100*n/total))

        n+=1

  """ Save a model to a path, returns the path to which it was saved """
  def saveModel(self, path):
    tf.add_to_collection('logits', self._logits)
    tf.add_to_collection('input', self._inpt)
    tf.add_to_collection('targets', self._targets)
    tf.add_to_collection('cross_entropy', self._cross_entropy)
    saver = tf.train.Saver()

    with open("%s.dict"%path, "wb") as f:
      pickle.dump(self._vocab, f)
    with open("%s.unk"%path, "wb") as f:
      pickle.dump(self.unker, f)

    p = saver.save(self.sess, path)
    return p # the filename under which the model was saved

  def generateClickbait(self, n):
    stopcode = self._vocab[STOP]
    #sentence = [ np.random.randint(vocab_size) ]
    sentence = [ stopcode ]
    while len(sentence) < n:
      words = sentence + [0]*(max_length - len(sentence)) # pad with zeros
      print(words)
      dist = np.array(self._probs.eval(feed_dict={self._inpt: [words]}, session=self._sess)[0])
      dist /= dist.sum()
      nword = np.random.choice(len(dist),p=dist)

      if nword == stopcode:
        break
      elif nword == 0:
        continue
      sentence.append(nword)

    s = sentence[1:] # exclude stop symbol
    return "".join([self._inv_map.get(w, "")+" " for w in s]).strip()

  def evaluateTitle(self, title):
    pass

  def _padList(self, l):
    return l + [0]*(max_length - len(l))

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
      text = [("%s %s %s"%(STOP,line,STOP)).split() for line in f]

    counts = {} # map word->count
    for l in text:
      for w in l:
        counts[w] = counts.get(w, 0) + 1 # increment count of this word

    self.unker = unk.BasicUnker(text, counts) # unk anything with count <= 1
    corpus = unker.getUnkedCorpus() # does what it says

    return corpus

