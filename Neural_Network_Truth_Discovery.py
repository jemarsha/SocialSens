from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

f = open('BrexitSensingMatrix.txt','r')
dim1 = 0
dim2 = 0
for line in f:
    tokens = line.split(',')
    if int(tokens[0]) > dim1:
        dim1 = int(tokens[0])
    if int(tokens[1]) > dim2:
        dim2 = int(tokens[1])
f.close()
dim1 = dim1
dim2 = dim2
social_matrix = np.zeros((dim1, dim2))

f = open('BrexitSensingMatrix.txt','r')
for line in f:
    tokens = line.split(',')
    social_matrix[int(tokens[0])-1][int(tokens[1])-1] = 1
f.close()

gold = np.zeros(dim2)
f = open('brexitGroundT.txt','r')
for line in f:
    tokens = line.split(',')
    gold[(int(tokens[0])-1)]=int(tokens[1])
f.close()

gold2 = np.zeros((dim2,2))
for x in range(0,len(social_matrix[0])):
	if gold[x] == 1:
		gold2[x][1] = 1
	else:
		gold2[x][0] = 0

gold_labels = np.zeros((dim1,2))
for x in range(0,len(social_matrix)):
    count_correct = 0
    count_total = 0
    for y in range(0,len(social_matrix[x])):
        if social_matrix[x][y] == 1:
            if gold[y] == 1:
                count_correct = count_correct + 1
                count_total = count_total +1
            else:
                count_total = count_total +1
    the_label = float(count_correct)/float(count_total)
    if the_label >= .75:
        gold_labels[x][1] = 1
    else:
        gold_labels[x][0] = 1

print(dim1)
print(dim2)
train_dataset = social_matrix[:,10:100].astype(np.float32)
train_labels = gold2[10:100].astype(np.float32)
valid_dataset = social_matrix[:,100:200].astype(np.float32)
valid_labels = gold2[100:200].astype(np.float32)
test_dataset = social_matrix[:,200:300].astype(np.float32)
test_labels = gold2[200:300].astype(np.float32)


train_dataset = train_dataset.transpose()
valid_dataset = valid_dataset.transpose()
test_dataset = test_dataset.transpose()

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_labels = 2

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into graph constants
  tf_train_dataset = tf.constant(train_dataset)
  tf_train_labels = tf.constant(train_labels)
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  # The weight matrices will be initialized using random valued following a (truncated)
  # Both biases get initialized to zero.


  #  weights_1 = tf.Variable(
  #       tf.truncated_normal([dim1, num_labels])
  #   )
  #biases_1 = tf.Variable(tf.zeros([num_labels]))



  #weights_2 = tf.Variable(tf.truncated_normal([2, 2]))
  #biases_2 = tf.Variable(tf.zeros([num_labels]))

  weights_1 = tf.Variable(
         tf.truncated_normal([dim1, 50])
     )
  biases_1 = tf.Variable(tf.zeros([50]))


  weights_2 = tf.Variable(tf.truncated_normal([50, num_labels]))
  biases_2 = tf.Variable(tf.zeros([num_labels]))

  weights_3 =  tf.Variable(tf.truncated_normal([num_labels, num_labels]))
  biases_3 = tf.Variable(tf.zeros([num_labels]))

  #weights_4= tf.Variable(tf.truncated_normal([num_labels, num_labels]))
 # biases_4= tf.Variable(tf.zeros([num_labels]))
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. 
  # Then, we use a RELU to pass to a second weight bias multiplication
  # We pass the result to compute the softmax and cross-entropy
  # We take the average of this cross-entropy across all training examples: that's our loss.
  #logits = tf.matmul(tf_train_dataset, weights_1) + biases_1
  h_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
  logits2 = tf.nn.relu(tf.matmul(h_1, weights_2) + biases_2)
  logits3 = tf.matmul(logits2, weights_3) + biases_3
  #logits4 = tf.nn.relu(tf.matmul(logits3, weights_4) + biases_4)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits3, tf_train_labels))
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  
  #train_prediction = tf.nn.softmax(logits2)
  #valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1),weights_2)+ biases_2)
  #test_prediction =  tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1),weights_2)+ biases_2)

  train_prediction = tf.nn.softmax(logits3)
  valid_prediction = tf.nn.softmax( tf.matmul(tf.nn.relu( tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1),weights_2)+ biases_2) ,weights_3) + biases_3)
  test_prediction =  tf.nn.softmax( tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1),weights_2)+ biases_2) ,weights_3) + biases_3)
num_steps = 4001

def accuracy(predictions, labels):
  the_accuracy = 0
  true_positives = 0
  false_positives = 0
  false_negatives = 0
  precision = 0
  recall = 0
  #print(predictions[0])
  for f in range(0,len(predictions)):
      predictions_f = np.argmax(predictions[f], 0)
      labels_f = np.argmax(labels[f], 0)
      if predictions_f == 1 and labels_f == 1:
       	true_positives = true_positives + 1
      elif predictions_f == 0 and labels_f == 1:
       	false_negatives = false_negatives + 1
      elif predictions_f == 1 and labels_f == 0:
       	false_positives = false_positives + 1
  #print(true_positives)
  #print(false_positives)
  #print(false_negatives)
  #print(len(predictions))
  #print("----")
  if (true_positives + false_positives) > 0:
  	precision = 100* (true_positives / float(true_positives + false_positives))
  if (true_positives + false_negatives) > 0:
  	recall = 100*(true_positives / float(true_positives + false_negatives))
  f_1 = 100*((precision + recall)/2)
  accurac = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
  return precision,recall,f_1,accurac

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      train_res = accuracy(predictions, train_labels)
      print('Training precision: %.1f Recall: %.1f F_1:%.1f Accuracy:%.1f' % (train_res[0],train_res[1],train_res[2],train_res[3]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      val_res = accuracy(valid_prediction.eval(), valid_labels)
      print('Validation precision: %.1f Recall: %.1f F_1:%.1f Accuracy:%.1f' %( val_res[0],val_res[1],val_res[2],val_res[3]))
  test_res = accuracy(test_prediction.eval(), test_labels)
  print('Test precision: %.1f Recall: %.1f F_1:%.1f Accuracy:%.1f' % (test_res[0],test_res[1],test_res[2],test_res[3]) )
