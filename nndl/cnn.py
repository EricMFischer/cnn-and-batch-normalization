import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_layers = 1 + hidden_dim
    '''
    input_dim: (3,16,16)
    num_filters: 3
    filter_size: 3
    hidden_dim: 7
    num_classes: 10
    weight_scale: 1e-3
    reg: 0.0
    use_batchnorm: False
    '''
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    C, H, W = input_dim
    # conv - relu - 2x2 max pool layer
    # conv layers accept weights shaped (F, C, HH, WW)
    conv_weight_size = (num_filters, C, filter_size, filter_size)
    self.params['W1'] = np.random.normal(0, weight_scale, conv_weight_size)
    self.params['b1'] = np.zeros(num_filters)
    
    # pool_layer_h calculation for next affine layer
    S = 1
    P = (filter_size - 1) / 2
    conv_layer_h = 1 + (H - filter_size + 2*P) / S
    S_pool = 2
    pool_h = 2
    pool_layer_h = int(1 + (conv_layer_h - pool_h) / S_pool)
    
    # affine - relu layer
    # affine layers accept weights shaped (input_dim, output_dim)
    affine_weight_size = (num_filters * pool_layer_h**2, hidden_dim)
    self.params['W2'] = np.random.normal(0, weight_scale, affine_weight_size)
    self.params['b2'] = np.zeros(hidden_dim)
    
    # affine - softmax layer
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    # X: (2,3,16,16)
    scores = None
    
    # conv - relu - 2x2 max pool layer
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    # affine - relu layer
    # affine layers accept x shaped (N, D)
    N, F, conv_h, conv_w = conv_out.shape
    conv_out = conv_out.reshape(N, F*conv_h*conv_w)
    affine_out, affine_cache = affine_relu_forward(conv_out, W2, b2)
    
    # affine - softmax layer
    scores, scores_cache = affine_forward(affine_out, W3, b3)

    if y is None:
      return scores
    
    loss = 0
    grads = {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y)
    for W in [W1, W2, W3]:
        loss += 0.5 * self.reg * np.sum(W**2)
    
    # backpropagation for affine - softmax layer
    dx3, dw3, db3 = affine_backward(dscores, scores_cache) # upstream deriv, cache
    grads['W3'] = dw3 + self.reg*W3
    grads['b3'] = db3
    
    # backward propagation for affine - relu layer
    dx2, dw2, db2 = affine_relu_backward(dx3, affine_cache)
    grads['W2'] = dw2 + self.reg*W2
    grads['b2'] = db2
    
    # backward propagation for conv - relu - 2x2 max pool layer
    dx2 = dx2.reshape(N, F, conv_h, conv_w) # reshape dx2 for conv layer backprop
    dx1, dw1, db1 = conv_relu_pool_backward(dx2, conv_cache)
    grads['W1'] = dw1 + self.reg*W1
    grads['b1'] = db1

    return loss, grads
