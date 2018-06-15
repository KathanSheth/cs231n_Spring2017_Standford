import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train,dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in range(num_train):
    correct_class = y[i]
    score = X[i].dot(W)
    score -= np.max(score)
    unnormalized_prob = np.exp(score)
    correct_class_prob = np.exp(score[y[i]])
    sum_prob = np.sum(unnormalized_prob)
    loss += -np.log(correct_class_prob/sum_prob)

    for j in range(num_classes):
      soft_score = np.exp(score[j])/sum_prob

      
      if j == correct_class:
        dW[:,j] += (soft_score - 1) * X[i]
      else:
        dW[:,j] += soft_score * X[i]




  loss = loss/num_train
  loss = loss + 0.5 * reg * np.sum(W * W)

  dW = dW / num_train
  dW += 2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train,dim = X.shape
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  #soft_score = np.exp(score) / np.sum(np.exp(score),axis=1)
  score -= np.max(score, axis=1)[:,None]
  correct_class_prob = np.exp(score[np.arange(num_train),y])
  sum_prob = np.sum(np.exp(score),axis=1)
  #print (correct_class_prob.shape)
  #print (sum_prob.shape)
  loss = -np.sum(np.log(correct_class_prob/sum_prob))

  loss /= num_train

  loss += 0.5 * reg * np.sum(W * W)

  #Below first term is 500X10 and second is (500,). When I divided it didnot work
  #Here I have to divide each raw element by a 1D vector so I need to use [:,None] or np.newaxes
  #Here is the link
  #https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
  soft_score = np.exp(score) / np.sum(np.exp(score),axis=1)[:,None]
  dScore = soft_score
  dScore[np.arange(num_train),y] = dScore[np.arange(num_train),y] - 1

  #print (X.shape)
  #print (dScore.shape)
  dW = X.T.dot(dScore)
  dW = dW / num_train
  dW += 2*reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

