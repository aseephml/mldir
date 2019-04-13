# =========================  NN 
import matplotlib.pyplot as plt
import tensorflow as tf

FEATURE_DIM = 561
LEARNING_RATE = 0.01
LABEL_DIM = 6
BATCH_SIZE = 64
NUM_EPOCH = 50

class Neural_Network():
  
  def __init__(self, feature_dim = FEATURE_DIM, label_dim = LABEL_DIM):
    self.feature_dim = feature_dim
    self.label_dim = label_dim
        
  def build_network(self, learning_rate=LEARNING_RATE):
    
    self.train_X = tf.placeholder(tf.float32, [None, self.feature_dim])
    self.train_Y = tf.placeholder(tf.float32, [None, self.label_dim])
    
    self.layer_1 = self.dense_layer(self.train_X, self.feature_dim, 
                                    1024, activation=tf.nn.relu, name='layer_1')
    self.layer_2 = self.dense_layer(self.layer_1, 1024, 512, 
                                   activation=tf.nn.relu, name='layer_2')
    self.layer_3 = self.dense_layer(self.layer_2, 512, 64, 
                                   activation=tf.nn.relu, name='layer_3')
    self.output = self.dense_layer(self.layer_3, 64, 6, name='output')
    
    self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels = self.train_Y))
    
    self.optimizer = tf.train.AdamOptimizer(learning_rate)
    
    self.train_step = self.optimizer.minimize(self.loss)
    
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,1), 
                                                    tf.argmax(self.train_Y, 1)),'float'))
    
  def dense_layer(self, inputs, input_size, output_size, name, activation=None):
    
    W = tf.get_variable(name=name+'_w',shape=(input_size, output_size), 
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name=name+'_b', shape=(output_size))
    out = tf.matmul(inputs, W) + b
    
    if activation:
      return activation(out)
    else:
      return out

# ========================= Data Object
class Data():
  
  def __init__(self, train_X, train_Y, batch_size=BATCH_SIZE):
    
    self.train_X = train_X
    self.train_Y = train_Y
    self.batch_size = batch_size
    self.num_batch = self.train_X.shape[0]//batch_size
    
  def generate_batch(self):
    
    for i in range(self.num_batch):
      
      x = self.train_X[(i*self.batch_size):(i+1)*self.batch_size, :]
      y = self.train_Y[(i*self.batch_size):(i+1)*self.batch_size]
      
      yield x, y 
      
# ----------------------- Training Object
class Learn():
  
  def __init__(self, train_X, train_Y, test_X, test_Y, 
               batch_size=BATCH_SIZE, epoch = NUM_EPOCH):
    
    self.batch_size = batch_size
    self.epoch = epoch
    
    self.network = Neural_Network()
    self.network.build_network(learning_rate=0.001)
    self.data = Data(train_X, train_Y, self.batch_size)
    self.test_X = test_X
    self.test_Y = test_Y
  
  def run_training(self):
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
      
      sess.run(init)
      
      training_loss = []
      counter, tmp_loss = 0, 0
      
      for i in range(self.epoch):
        
        for x, y in self.data.generate_batch():
          
          feed_dict = {self.network.train_X:x, self.network.train_Y:y}
        
          _, loss = sess.run([self.network.train_step, self.network.loss], 
                             feed_dict=feed_dict)
          
          if counter % 100 == 0 and counter!=0:
            training_loss.append(tmp_loss/100)
            tmp_loss = 0
          else:
            tmp_loss += loss
            
          counter += 1
          
        print("Epoch {0}, loss is {1}".format(i, loss))
        
      self.training_loss = training_loss
      acc = sess.run([self.network.accuracy], feed_dict={self.network.train_X:self.test_X,
                                                        self.network.train_Y:self.test_Y})
      print("The testing accuracy is {0}".format(acc))
      
  def plot_training_loss(self):
    plt.plot(self.training_loss)      