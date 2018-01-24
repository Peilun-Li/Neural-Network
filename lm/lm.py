import numpy as np
import argparse
from collections import defaultdict

class Graph:
  def __init__(self):
    self.layers = []
    self.optimizer = None
    self.phase = "Train"

  def forward(self):
    for layer in self.layers:
      layer.forward()

  def backward(self):
    for layer in self.layers[::-1]:
      layer.backward()

  def renew_cg(self):
    for layer in self.layers:
      del layer
    self.layers = []

  def clear_tensor(self):
    for layer in self.layers:
      layer.clear_tensor()

  def clear_epochwise(self):
    for layer in self.layers:
      layer.clear_epochwise()

graph = Graph()

class SGD:
  def __init__(self, lr=0.1, momentum=0.9, weight_decay=0.0000):
    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.deri_dict = {}

  def update(self, val, deri, decay_weight=False):
    if decay_weight:
      val *= (1.0 - self.weight_decay)
    if id(val) not in self.deri_dict:
      self.deri_dict[id(val)] = np.copy(val)
    else:
      self.deri_dict[id(val)] = self.deri_dict[id(val)] * self.momentum + deri
    val -= self.lr * self.deri_dict[id(val)]
    #val -= self.lr * deri
    #return val - self.lr * deri

class Layer(object):
  def __init__(self):
    self.next_layer = None
    self.prev_layer = None
    self.size = None

    self.input_val = None
    self.output_val = None
    self.back_deri = None

  def forward(self):
    pass

  def backward(self):
    pass

  def clear_tensor(self):
    self.input_val, self.output_val, self.back_deri = None, None, None

  def clear_epochwise(self):
    pass

class RBM(Layer):
  def __init__(self, prev, size, K):
    super(RBM, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = size
    self.K = K

    # W: [hidden state dim, input dim]
    self.W = np.random.normal(loc=0.0, scale=0.1, size=(self.size, self.prev_layer.size))
    # b: [1, hidden state dim]
    self.b = np.zeros((1, self.size))
    # c: [1, input dim]
    self.c = np.zeros((1, self.prev_layer.size))

    # input shape: [batch, self.prev_layer.size]
    # output shape: [batch, self.prev_layer.size] CD_k
    # back_deri: None
    graph.layers.append(self)

  def set_label(self, y):
    pass

  def forward(self):
    #CD_k
    # [batch, input dim]
    self.sample_x = hard_cut(self.input_val)
    for k in range(self.K):
      #[batch, hidden state dim]
      self.p_hx = sigmoid(np.matmul(self.sample_x, np.transpose(self.W)) + self.b)
      #[batch, hidden state dim]
      self.sample_h = sample(self.p_hx)
      #[batch, input dim]
      self.p_xh = sigmoid(np.matmul(self.sample_h, self.W) + self.c)
      self.sample_x = sample(self.p_xh)
    # cross entropy loss
    input_x = hard_cut(self.input_val)
    self.loss = input_x * np.log(self.p_xh) + (1 - input_x) * np.log(1-self.p_xh)
    self.loss = -1.0 * np.sum(self.loss, axis=1, keepdims=True)
    self.correct = 0

  def backward(self):
    # [batch, hidden state dim]
    self.h_x = sigmoid(np.matmul(self.input_val, np.transpose(self.W)) + self.b)
    self.h_x_tilda = sigmoid(np.matmul(self.sample_x, np.transpose(self.W)) + self.b)
    # [hidden state dim, input dim]
    self.W_deri = -1.0 * (np.matmul(np.transpose(self.h_x), self.input_val) - np.matmul(np.transpose(self.h_x_tilda), self.sample_x)) / self.input_val.shape[0]
    # [1, hidden state dim]
    self.b_deri = -1.0 * np.mean(self.h_x - self.h_x_tilda, axis=0, keepdims=True)
    # [1, input dim]
    self.c_deri = -1.0 * np.mean(self.input_val - self.sample_x, axis=0, keepdims=True)
    
    graph.optimizer.update(self.W, self.W_deri)
    graph.optimizer.update(self.b, self.b_deri)
    graph.optimizer.update(self.c, self.c_deri)
    

def sample(i):
  rand = np.random.rand(*i.shape)
  return (i>rand).astype(np.float)

def hard_cut(i):
  o = np.copy(i)
  o[o>=0.5] = 1
  o[o!=1] = 0
  return o

class AutoEncoder(Layer):
  def __init__(self, prev, size, noise):
    super(AutoEncoder, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = size
    self.noise = noise

    # [hidden state dim, input dim]
    self.W = np.random.normal(loc=0.0, scale=0.1, size=(self.size, self.prev_layer.size))

    # input shape: [batch, input dim]
    # output shape: [batch, input dim]
    # back_deri: [batch, input dim]
    graph.layers.append(self)

  def forward(self):
    # [batch, hidden state dim]
    self.noise_mask = np.random.rand(*self.input_val.shape)
    self.noise_mask = (self.noise_mask>self.noise).astype(np.float)
    self.noise_input = self.noise_mask * self.input_val
    self.z = sigmoid(np.matmul(self.noise_input, np.transpose(self.W)))
    # [batch, input dim]
    self.output_val = sigmoid(np.matmul(self.z, self.W))
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    # [batch, input dim]
    self.sig2_deri = self.back_deri * self.output_val * (1.0 - self.output_val)
    # [hidden state dim, input dim]
    self.W_deri = np.matmul(np.transpose(self.z), self.sig2_deri) / self.output_val.shape[0]
    # [batch, hidden state dim]
    self.z_deri = np.matmul(self.sig2_deri, np.transpose(self.W))
    # [batch, hidden state dim]
    self.sig1_deri = self.z_deri * self.z * (1.0 - self.z)
    # [hidden state dim, input dim]
    self.W_deri += np.matmul(np.transpose(self.sig1_deri), self.noise_input) / self.input_val.shape[0]
    if self.prev_layer and self.prev_layer.back_deri is not None:
      # [batch, input dim]
      self.prev_layer.back_deri += np.matmul(self.sig1_deri, self.W)

    graph.optimizer.update(self.W, self.W_deri)

class Dense(Layer):
  def __init__(self, prev, size):
    super(Dense, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = size
    
    uniform_b = np.sqrt(6) / np.sqrt(self.size + self.prev_layer.size)
    # W: [self.prev_layer.size, self.size]
    self.W = np.random.uniform(low=-uniform_b, high=uniform_b, size=(self.prev_layer.size, self.size))
    # b: [1, self.size]
    self.b = np.zeros((1, self.size))
    
    # input shape: [batch, self.prev_layer.size]
    # output shape: [batch, self.size]
    # back_deri shape: [batch, self.size]
    graph.layers.append(self)
  
  def forward(self):
    self.output_val = np.matmul(self.input_val, self.W) + self.b
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    # [self.prev_layer.size, self.size]
    self.W_deri = np.matmul(np.transpose(self.input_val), self.back_deri) / self.back_deri.shape[0]
    # [1, self.size]
    self.b_deri = np.mean(self.back_deri, axis=0, keepdims=True)

    if self.prev_layer and self.prev_layer.back_deri is not None:
      # [batch, self.prev_layer.size]
      self.prev_layer.back_deri += np.matmul(self.back_deri, np.transpose(self.W))

    graph.optimizer.update(self.W, self.W_deri, decay_weight=True)
    graph.optimizer.update(self.b, self.b_deri)

class BatchNorm(Layer):
  def __init__(self, prev):
    super(BatchNorm, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = self.prev_layer.size

    # [1, size]
    self.gamma = np.ones((1, self.size))
    # [1, size]
    self.beta = np.zeros((1, self.size))
    self.x_epoch = None
    self.eps = 1e-8

    # input shape: [batch, self.size]
    # output shape: [batch, self.size]
    # back_deri shape: [batch, self.size]
    graph.layers.append(self)

  def forward(self):
    # [1, size]
    self.mu = np.mean(self.input_val, axis=0, keepdims=True)
    # [1, size]
    self.var = np.var(self.input_val, axis=0, keepdims=True)
    if graph.phase == "train":
      if self.x_epoch is None:
        self.x_epoch = np.copy(self.input_val)
      else:
        self.x_epoch = np.concatenate((self.x_epoch, self.input_val))
    else:
      self.mu = np.mean(self.x_epoch, axis=0, keepdims=True)
      self.var = np.var(self.x_epoch, axis=0, keepdims=True)

    self.x_mu = self.input_val - self.mu
    self.inv_var = 1.0 / np.sqrt(self.var + self.eps)
    self.x_hat = self.x_mu * self.inv_var
    self.output_val = self.gamma * self.x_hat + self.beta
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    N, D = self.back_deri.shape
    x_hat_deri = self.back_deri * self.gamma
    var_deri = np.sum((x_hat_deri * self.x_mu * (-0.5) * (self.inv_var)**3), axis=0, keepdims=True)
    mu_deri = (np.sum((x_hat_deri * -self.inv_var), axis=0, keepdims=True)) + (var_deri * (-2.0 / N) * np.sum(self.x_mu, axis=0, keepdims=True))
    if self.prev_layer and self.prev_layer.back_deri is not None:
      self.prev_layer.back_deri += x_hat_deri * self.inv_var + var_deri * (2.0 / N) * self.x_mu + (1.0 / N) * mu_deri

    beta_deri = np.sum(self.back_deri, axis=0, keepdims=True)
    gamma_deri = np.sum(self.x_hat*self.back_deri, axis=0, keepdims=True)

    graph.optimizer.update(self.gamma, gamma_deri)
    graph.optimizer.update(self.beta, beta_deri)

  def clear_epochwise(self):
    self.x_epoch = None
   
class ImageCrossEntropyLoss(Layer):
  def __init__(self, prev):
    super(ImageCrossEntropyLoss, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = None
    graph.layers.append(self)

  def set_label(self, y):
    # [batch, input_dim]
    self.truth = hard_cut(y)

  def forward(self):
    # [batch, input_dim]
    self.predict = self.input_val
    self.loss = self.truth * np.log(self.predict) + (1 - self.truth) * np.log(1-self.predict)
    # [batch, 1]
    self.loss = -1.0 * np.sum(self.loss, axis=1, keepdims=True)
    self.correct = 0

  def backward(self):
    # [batch, input_dim]
    if self.prev_layer and self.prev_layer.back_deri is not None:
      self.prev_layer.back_deri += (self.predict - self.truth) / (self.predict * (1 - self.predict))
  

class SoftmaxWithCrossEntropyLoss(Layer):
  def __init__(self, prev):
    super(SoftmaxWithCrossEntropyLoss, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = self.prev_layer.size
    graph.layers.append(self)
 
  def set_label(self, y):
    #[batch, 1]
    self.y = y
    
  def forward(self):
    #[batch, size]
    self.softmax = np.exp(self.input_val)
    self.softmax = self.softmax / np.sum(self.softmax, axis=1, keepdims=True)
    # [batch, 1]
    self.prediction = np.argmax(self.softmax, axis=1).reshape((-1, 1))
    self.correct = np.sum(self.y == self.prediction)

    # [batch, 1]
    self.loss = np.array(map(lambda k: np.take(k[0], k[1]), zip(self.softmax, self.y)))
    self.loss = -1.0 * np.log(self.loss)

  def backward(self):
    # [batch, size]
    y_onehot = np.eye(self.size)[self.y.reshape(-1)]
    if self.prev_layer and self.prev_layer.back_deri is not None:
      self.prev_layer.back_deri += -1.0 * (y_onehot - self.softmax)
    
  def clear_tensor(self):
    super(SoftmaxWithCrossEntropyLoss, self).clear_tensor()
    self.softmax, self.prediction, self.loss = None, None, None
    self.correct = 0

def sigmoid(i):
  return 1.0 / (1.0 + np.exp(-1.0 * i))

class Sigmoid(Layer):
  def __init__(self, prev):
    super(Sigmoid, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = self.prev_layer.size

    graph.layers.append(self)

  def forward(self):
    #[batch, size]
    self.output_val = np.copy(self.input_val)
    self.output_val = 1.0 / (1.0 + np.exp(-1.0 * self.output_val))
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    if self.prev_layer and self.prev_layer.back_deri is not None:
      self.prev_layer.back_deri += self.back_deri * self.output_val * (1.0 - self.output_val)

class Tanh(Layer):
  def __init__(self, prev):
    super(Tanh, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = self.prev_layer.size

    graph.layers.append(self)

  def forward(self):
    #[batch, size]
    self.output_val = np.copy(self.input_val)
    self.output_val = (np.exp(2 * self.output_val) - 1) / (np.exp(2 * self.output_val) + 1)
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    if self.prev_layer and self.prev_layer.back_deri is not None:
      self.prev_layer.back_deri += self.back_deri * (1 - self.output_val * self.output_val)


class ReLU(Layer):
  def __init__(self, prev):
    super(ReLU, self).__init__()
    self.prev_layer = prev
    self.prev_layer.next_layer = self
    self.size = self.prev_layer.size

    graph.layers.append(self)

  def forward(self):
    #[batch, size]
    self.output_val = np.copy(self.input_val)
    self.output_val = np.maximum(self.output_val, 0.0)
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    if self.prev_layer and self.prev_layer.back_deri is not None:
      self.prev_layer.back_deri += self.back_deri * (self.input_val > 0.0)


class Input(Layer):
  def __init__(self, size):
    super(Input, self).__init__()
    self.size = size
    graph.layers.append(self)

  def set_input(self, array):
    #[batch, size]
    if array.shape[1] != self.size:
      print "invalid input size"
      return
    self.output_val = np.copy(array)

  def forward(self):
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

class EmbeddingInput(Layer):
  def __init__(self, vocabulary, embedding_dimension=16, n_gram=4):
    super(EmbeddingInput, self).__init__()
    self.concat_cell = n_gram - 1
    self.embed_dim = embedding_dimension
    self.voc = vocabulary
    self.V = len(self.voc)
    self.size = self.concat_cell * self.embed_dim
    # [V, embed_dim]
    self.W = np.random.uniform(size=(self.V, self.embed_dim))
    graph.layers.append(self)

  def set_input(self, array):
    # array shape: [batch, n_gram-1]
    # output shape: [batch, (n_gram-1) * self.embed_dim]
    self.indices = np.copy(array)
    self.output_val = np.zeros((array.shape[0], self.embed_dim * self.concat_cell))
    for i in range(array.shape[0]):
      for j in range(array.shape[1]):
        self.output_val[i,j*self.embed_dim:(j+1)*self.embed_dim] = self.W[self.indices[i,j],:]

  def forward(self):
    self.back_deri = np.zeros_like(self.output_val)
    if self.next_layer:
      self.next_layer.input_val = np.copy(self.output_val)

  def backward(self):
    W_deri = np.zeros_like(self.W)
    for i in range(self.indices.shape[0]):
      for j in range(self.concat_cell):
        idx = self.indices[i][j]
        W_deri[idx,:] += self.back_deri[i,j*self.embed_dim:(j+1)*self.embed_dim]
    graph.optimizer.update(self.W, W_deri)

def load_MNIST(path):
  X, Y = [], []
  with open(path) as f:
    for line in f:
      line = map(float,line.strip().split(","))
      X.append(line[:-1])
      Y.append(line[-1:])
  return np.array(X), np.array(Y, dtype=np.int64)

def load_lm(output_statistics=True):
  voc = defaultdict(int)
  train_sentences = []
  with open('train.txt') as f:
    for line in f:
      line = ["START"] + line.strip().lower().split(" ") + ["END"]
      train_sentences.append(line[:])
      for word in line:
        voc[word] += 1
  sorted_wordcnt = sorted(voc.items(), key=lambda k:k[1], reverse=True)
  #print len(voc)
  #print sorted_wordcnt[:100]
  voc = {"UNK":0}
  for k, v in sorted_wordcnt[:7999]:
    voc[k] = v
  for i in range(len(train_sentences)):
    for j in range(len(train_sentences[i])):
      if train_sentences[i][j] not in voc:
        voc["UNK"] += 1
        train_sentences[i][j] = "UNK"
  #print len(voc)
  #print sorted(voc.items(), key=lambda k:k[1], reverse=True)[:100]
  #print len(train_sentences)
  #print train_sentences[:10]

  idx = 0
  for k, v in voc.items():
    voc[k] = idx
    idx += 1

  val_sentences = []
  with open('val.txt') as f:
    for line in f:
      line = ["START"] + line.strip().lower().split(" ") + ["END"]
      for i in range(len(line)):
        if line[i] not in voc:
          line[i] = "UNK"
      val_sentences.append(line[:])
  #print len(val_sentences)
  #print val_sentences[:10]

  if output_statistics:
    dic_4gram = defaultdict(int)
    for i in range(len(train_sentences)):
      for j in range(4, len(train_sentences[i])+1):
        dic_4gram[" ".join(train_sentences[i][j-4:j])] += 1
    dic_4gram_freqcnt = defaultdict(int)
    for k, v in dic_4gram.items():
      dic_4gram_freqcnt[v] += 1
    print "TOP 50 common 4-grams"
    for k, v in sorted(dic_4gram.items(), key=lambda k:k[1], reverse=True)[:50]:
      print k,"frequency", v
    print "freq - cnt"
    for k, v in dic_4gram_freqcnt.items():
      print k, "-", v
    import matplotlib.pyplot as plt
    l = plt.hist(dic_4gram.values(), bins=42, align="left")
    plt.xlabel("frequency")
    plt.ylabel("count")

    plt.show()

  for i in range(len(train_sentences)):
    for j in range(len(train_sentences[i])):
      train_sentences[i][j] = voc[train_sentences[i][j]]
  for i in range(len(val_sentences)):
    for j in range(len(val_sentences[i])):
      val_sentences[i][j] = voc[val_sentences[i][j]]
  return voc, train_sentences, val_sentences

def shuffle(input_X, input_Y):
  s = np.arange(input_X.shape[0])
  np.random.shuffle(s)
  return input_X[s], input_Y[s]

def sample_lm(voc):
  sentences = ["government of united", "city of new", "life in the", "he is the", "do you like"]
  sent_to_idx = []
  for i in range(len(sentences)):
    words = sentences[i].split()
    for j in range(len(words)):
      words[j] = voc[words[j]]
    sent_to_idx.append(words[:])
  for i in range(len(sentences)):
    for j in range(10):
      graph.layers[0].set_input(np.array([sent_to_idx[i][-3:]]))
      graph.layers[-1].set_label(np.array([[0]]))
      graph.forward()
      next_word_idx = graph.layers[-1].prediction[0][0]
      sent_to_idx[i].append(next_word_idx)
      next_word = ""
      for k, v in voc.items():
        if v == next_word_idx:
          next_word = k
          break
      sentences[i] = sentences[i] + " " + next_word
  print sentences
          
def train():
  global train_X, train_Y
  train_X, train_Y = shuffle(train_X, train_Y)
  res = []
  for epoch in range(num_epoch):
    graph.phase = "train"
    loss, accuracy = 0.0, 0.0
    for i in range(len(train_X)/batch_size):
      graph.layers[0].set_input(train_X[i*batch_size:(i+1)*batch_size])
      # AE uses X as label
      if type(graph.layers[-1]) is ImageCrossEntropyLoss:
        graph.layers[-1].set_label(train_X[i*batch_size:(i+1)*batch_size])
      else:
        graph.layers[-1].set_label(train_Y[i*batch_size:(i+1)*batch_size])
      graph.forward()
      loss += np.sum(graph.layers[-1].loss)
      accuracy += graph.layers[-1].correct
      graph.backward()
      graph.clear_tensor()
    loss /= ((len(train_X) / batch_size) * batch_size)
    accuracy /= ((len(train_X) / batch_size) * batch_size)
    valid_loss, valid_accuracy = validation(valid_X, valid_Y)
    test_loss, test_accuracy = validation(test_X, test_Y)
    graph.clear_epochwise()
    # different log format for LM
    if type(graph.layers[0]) is EmbeddingInput:
      print "Epoch %d: train (loss %.6f, perplexity %.6f), valid (loss %.6f, perplexity %.6f)" % (epoch+1, loss, np.exp(loss), valid_loss, np.exp(valid_loss))
      res.append((epoch+1, loss, np.exp(loss), valid_loss, np.exp(valid_loss)))
    else:
      print "Epoch %d: train (loss %.6f, error %.6f%%), valid (loss %.6f, error %.6f%%), test (loss %.6f, error %.6f%%)" % (epoch+1, loss, (1-accuracy)*100, valid_loss, (1-valid_accuracy)*100, test_loss, (1-test_accuracy)*100)
      res.append((epoch+1, loss, (1-accuracy)*100, valid_loss, (1-valid_accuracy)*100, test_loss, (1-test_accuracy)*100))
  return res

def validation(input_X, input_Y):
  if input_X is None or input_Y is None:
    return float("inf"), float("inf")
  graph.phase = "valid"
  loss, accuracy = 0.0, 0.0
  for i in range(len(input_X)/batch_size):
    graph.layers[0].set_input(input_X[i*batch_size:(i+1)*batch_size])
    # AE uses X as label
    if type(graph.layers[-1]) is ImageCrossEntropyLoss:
      graph.layers[-1].set_label(input_X[i*batch_size:(i+1)*batch_size])
    else:
      graph.layers[-1].set_label(input_Y[i*batch_size:(i+1)*batch_size])
    graph.forward()
    loss += np.sum(graph.layers[-1].loss)
    accuracy += graph.layers[-1].correct
    graph.clear_tensor()
  loss /= ((len(input_X) / batch_size) * batch_size)
  accuracy /= ((len(input_X) / batch_size) * batch_size)
  return loss, accuracy

def plot_res(res, with_test=False):
  import matplotlib.pyplot as plt
  # [epoch, train_loss, train_error, valid_loss, valid_error, test_loss, test_error]
  res = np.array(res)
  plt.figure(1)
  #plt.subplot(211)
  plt.plot(res[:,0], res[:,1], 'b', label="train loss")
  plt.plot(res[:,0], res[:,3], 'r', label="valid loss")
  if with_test:
    plt.plot(res[:,0], res[:,5], 'g', label="test loss")
  plt.xlabel("epoch")
  plt.ylabel("average cross-entropy loss")
  plt.legend(loc='upper right')
  #plt.subplot(212)
  #plt.plot(res[:,0], res[:,2], 'b', label="train error")
  #plt.plot(res[:,0], res[:,4], 'r', label="valid error")
  #if with_test:
  #  plt.plot(res[:,0], res[:,6], 'g', label="test error")
  #plt.xlabel("epoch")
  #plt.ylabel("classification error %")
  #plt.legend(loc='upper right')
  plt.show()

def plot_res_lm(res, with_test=False):
  import matplotlib.pyplot as plt
  # [epoch, train_loss, train_perp, valid_loss, valid_perp]
  res = np.array(res)
  plt.figure(1)
  plt.subplot(211)
  plt.plot(res[:,0], res[:,1], 'b', label="train average loss")
  plt.plot(res[:,0], res[:,3], 'r', label="valid average loss")
  if with_test:
    plt.plot(res[:,0], res[:,5], 'g', label="test loss")
  plt.xlabel("epoch")
  plt.ylabel("average cross-entropy loss")
  plt.legend(loc='upper right')
  plt.subplot(212)
  plt.plot(res[:,0], res[:,2], 'b', label="train perplexity")
  plt.plot(res[:,0], res[:,4], 'r', label="valid perplexity")
  if with_test:
    plt.plot(res[:,0], res[:,6], 'g', label="test error")
  plt.xlabel("epoch")
  plt.ylabel("perplexity")
  plt.legend(loc='upper right')
  plt.show()

def similar_exp(voc):
  # [V, D]
  W = graph.layers[0].W
  words = ["he", "she", "car", "vehicle", "man", "woman", "japan", "japanese", "america", "american"]
  for i in range(len(words)):
    for j in range(i+1, len(words)):
      print words[i], words[j], np.linalg.norm(W[voc[words[i]],:] - W[voc[words[j]],:])

def vis_word_scatter_plot(voc):
  from random import shuffle
  import matplotlib.pyplot as plt
  word_idx_list = voc.items()
  shuffle(word_idx_list)
  x, y, words = [], [], []
  # [V, D]
  W = graph.layers[0].W
  for w, idx in word_idx_list[:500]:
    words.append(w)
    x.append(W[idx,0])
    y.append(W[idx,1])
  fig, ax = plt.subplots()
  ax.scatter(x,y,s=1)
  for i, word in enumerate(words):
    ax.annotate(word, (x[i],y[i]), size=20)

  plt.show()

def vis_layer(W):
  import matplotlib.pyplot as plt
  # W: [784, 100]
  plt.figure(1)
  for i in range(10):
    for j in range(10):
      idx = i * 10 + j
      plt.subplot(10, 10, idx + 1)
      plt.axis("off")
      plt.imshow(W[:,idx].reshape((28,28)), cmap="gray")
  plt.show()

def gibbs_sample():
  backup_k = graph.layers[1].K
  graph.layers[1].K = 1000
  #input_x = np.random.randint(0, 2, (100, 784)).astype(np.float)
  input_x = np.random.rand(100, 784)
  graph.layers[0].set_input(input_x)
  graph.forward()
  vis_layer(np.transpose(graph.layers[1].sample_x))
  graph.layers[1].K = backup_k
  graph.clear_tensor()

def prepare_lm_data(sentences, voc):
  data = []
  label = []
  for i in range(len(sentences)):
    for j in range(4,len(sentences[i])+1):
      data.append(sentences[i][j-4:j-1])
      label.append(sentences[i][j-1:j])
  return np.array(data), np.array(label)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", "-b", default=1000, type=int, help="batch size")
  parser.add_argument("--num_epoch", "-e", default=100, type=int, help="number of epochs")
  parser.add_argument("--momentum", "-m", default=0.0, type=float, help="momentum")
  parser.add_argument("--lr", "-l", default=0.1, type=float, help="learnign rate")
  parser.add_argument("--weight_decay", "-d", default=0.0000, type=float, help="weight decay (L2)")
  args = vars(parser.parse_args())

  voc, train_sentences, val_sentences = load_lm(output_statistics=False)
  train_X, train_Y = prepare_lm_data(train_sentences, voc)
  print train_X.shape
  valid_X, valid_Y = prepare_lm_data(val_sentences, voc)
  print valid_X.shape
  test_X, test_Y = None, None
  #train_X, train_Y = load_MNIST("digitstrain.txt")
  #valid_X, valid_Y = load_MNIST("digitsvalid.txt")
  #test_X, test_Y = load_MNIST("digitstest.txt")

  graph.renew_cg()
  sgd = SGD(args["lr"], args["momentum"], args["weight_decay"])
  graph.optimizer = sgd

  batch_size = args["batch_size"]
  num_epoch = args["num_epoch"]

  # LM
  l = EmbeddingInput(voc, embedding_dimension=16, n_gram=4)
  l = Dense(l, 128)
  l = Tanh(l)
  l = Dense(l, len(voc))
  l = SoftmaxWithCrossEntropyLoss(l)
  res = train()
  #plot_res_lm(res)
  #sample_lm(voc)
  #similar_exp(voc)
  #vis_word_scatter_plot(voc)

  #plot_res(res, with_test=True)
  #W = graph.layers[1].W
  #vis_layer(np.transpose(W))
  #gibbs_sample()
  #exit(0)

  #W_init = np.copy(graph.layers[1].W)
  #graph.renew_cg()
  #l = Input(784)
  #l = Dense(l, 100)
  #graph.layers[1].W = np.transpose(W_init)
  #l = Sigmoid(l)
  #l = Dense(l, 10)
  #l = SoftmaxWithCrossEntropyLoss(l)
  #res = train()

