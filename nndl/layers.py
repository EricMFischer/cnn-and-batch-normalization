import numpy as np
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

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  '''
  x: (2,4,5,6) (N,D) (num_inputs, input_shape/dimension tuple)
  w: (120,3) (D,M) (input_shape/dimension, output_dim) (np.prod(N), output_dim)
  b: (3,) (M) (output_dim)
  '''
  # ================================================================ #
  #   Calculate the output of the forward pass.  Notice the dimensions
  #   of w are D x M, which is the transpose of what we did in earlier 
  #   assignments.
  # ================================================================ #

  x_reshaped = x.reshape(x.shape[0], -1)
  out = x_reshaped.dot(w) + b # (num_inputs, output_dim) (N,M)

  cache = (x, w, b)
  return out, cache

def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  '''
  cache:
  x = np.random.randn(10,2,3) (N,D) (num_inputs, input_shape/dimension tuple)
  w = np.random.randn(6,5) (D,M) (input_shape/dimension, output_dim)
  b = np.random.randn(5) (M) (output_dim)
  
  dout = np.random.randn(10, 5) (N,M) (num_inputs, output_dim)
  '''
  x, w, b = cache

  # ================================================================ #
  #   Calculate the gradients for the backward pass.
  # ================================================================ #
  x_reshaped = x.reshape(x.shape[0], -1) # (10,6)
    
  # dx reshaping: remember each example x[i] has shape (d_1, ..., d_k)
  dx = dout.dot(w.T).reshape(x.shape) # (10,5).(5,6) = (10,6) -> (10,2,3)
  dw = x_reshaped.T.dot(dout) # (6,10).(10,5) = (6,5)
  db = np.sum(dout, axis=0).T

  '''
  dx: (10, 2, 3) (num_inputs, input_shape/dimension tuple)
  dw: (6, 5) (dimension, output_dim)
  db: (5,) (output_dim)
  '''
  return dx, dw, db

def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  # x: (3,4)
  out = np.maximum(0, x)

  cache = x
  return out, cache

def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  '''
  During forward propagation, only h hidden unit values larger than 0
  passed through the relu activation function np.maximum(0, x) before
  serving as x input values to Wx + b calculations in further layers.

  Thus, for backward propagation, only cached x input values (hidden
  unit ouput values during forward progagation) > 0 are multiplied by
  the upstream derivative dout to get dx, our gradients with respect to x.
  This makes sense because the hidden unit output values <= 0 did not
  "make it" to further layers and take part in any further calculations.
  '''
  # cache came from relu_forward
  # dout, cache: (10,10)
  x = cache 
  dx = (x > 0) * dout
  return dx

def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var: Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    # ================================================================ #
    # YOUR CODE HERE:
    #   A few steps here:
    #     (1) Calculate the running mean and variance of the minibatch.
    #     (2) Normalize the activations with the batch mean and variance.
    #     (3) Scale and shift the normalized activations.  Store this
    #         as the variable 'out'
    #     (4) Store any variables you may need for the backward pass in
    #         the 'cache' variable.
    # ================================================================ #
    
    u = x.mean(axis=0)
    x_minus_u = x - u
    var = np.mean(x_minus_u**2, axis=0)
    std = np.sqrt(var + eps)
    
    # store running mean and variance of the minibatches for each feature
    running_mean = momentum*u + (1 - momentum)*u
    running_var = momentum*var + (1 - momentum)*var
    # normalize activations with batch mean and variance
    xhat = x_minus_u / std
    # scale (gamma) and shift (beta) the normalized activations
    out = gamma*xhat + beta

    # store variables for backward pass
    cache = (gamma, mode, x_minus_u, std, xhat, out)
  
  elif mode == 'test':
    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the testing time normalized activations.  Normalize using
    #   the running mean and variance, and then scale and shift appropriately.
    #   Store the output as 'out'.
    # ================================================================ #
    
    # normalize activations with batch mean and variance
    std = np.sqrt(running_var + eps)
    xhat = (x - running_mean) / std
    # scale (gamma) and shift (beta) the normalized activations
    out = gamma*xhat + beta

    cache = (gamma, mode, std, xhat, out)
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # store updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D) (4,5)
  - cache: Variable of intermediates from batchnorm_forward. (10,)
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the batchnorm backward pass, calculating dx, dgamma, and dbeta.
  # ================================================================ #
  
  # gamma, beta, u, var, std:  scalars per feature (5,1)
  # dout, x, x_minus_u, xhat, out:  for all examples (4,5)
  # mode: 'train' or 'test'
  gamma, mode, x_minus_u, std, xhat, out = cache
  m = xhat.shape[0]

  dbeta = np.sum(dout, axis=0) # (5,)
  dgamma = np.sum(xhat * dout, axis=0) # (5,)

  dxhat = gamma * dout # (5,)*(4,5) = (4,5)
  da = dxhat / std
  dvardx = 2 * x_minus_u / m
  dvar = np.sum(-.5 * std**-3 * x_minus_u * dxhat, axis=0)
  dudx = 1 / m
  du = -np.sum(da, axis=0)

  dx = da + dvardx*dvar + dudx*du
  '''
  da: (4, 5)
  dvardx: (4, 5)
  dvar: (5,)
  dudx: scalar
  du: (5,)
  '''
  return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  if mode == 'train':
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the inverted dropout forward pass during training time.  
    #   Store the masked and scaled activations in out, and store the 
    #   dropout mask as the variable mask.
    # ================================================================ #

    # x: (500,500), p: 0.3 (drop each neuron with this prob)
    # dropping p=0.3 of neurons = keeping (1 - p) neurons
    mask = (np.random.rand(*x.shape) < (1 - p)) / (1 - p)
    out = x * mask # store the masked and scaled activations in out

  elif mode == 'test':
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the inverted dropout forward pass during test time.
    # ================================================================ #
    mask = None
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache

def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the inverted dropout backward pass during training time.
    # ================================================================ #
    dx = dout * mask

  elif mode == 'test':
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the inverted dropout backward pass during test time.
    # ================================================================ #
    dx = dout

  return dx

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx

def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
