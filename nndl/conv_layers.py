import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  pad = conv_param['pad'] # 1
  stride = conv_param['stride'] # 2

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  N, C, H, W = x.shape # (2,3,4,4)
  F, _, HH, WW = w.shape # (3,3,4,4)
  output_h = int(1 + (H + 2*pad - HH) / stride) # 2
  output_w = int(1 + (W + 2*pad - WW) / stride) # 2

  x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
  
  out = np.zeros((N, F, output_h, output_w)) # (2,3,2,2)
  for n in range(N):
    img = x_pad[n] # one image (C,H,W) (3,4,4)
    for f in range(F):
        filter = w[f] # one filter (C,HH,WW) (3,4,4)
        f_bias = b[f]
        for height in range(output_h):
            for width in range(output_w):
                hi, wi = height*stride, width*stride
                img_comp = img[:, hi:hi+HH, wi:wi+WW]
                out[n, f, height, width] = np.sum(img_comp*filter) + f_bias
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """

  N, F, output_h, output_w = dout.shape # 4,2,5,5
  x, w, b, conv_param = cache # (4,3,5,5) (2,3,3,3) (2,)
  _, C, H, W = x.shape # 4,3,5,5
  _, _, HH, WW = w.shape
  
  stride, pad = conv_param['stride'], conv_param['pad'] # 1, 1
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant') # (4,3,7,7)

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  
  dx_pad = np.zeros(xpad.shape) # (4,3,7,7)
  dw = np.zeros(w.shape)
  db = np.sum(dout, axis=(0,2,3))

  for dout_hi in range(output_h): # 5
    for dout_wi in range(output_w): # 5
        # dx_pad
        for n in range(N):
            # f_dout_vals: for all filters, one data point's dout scalar output
            n_dout_vals = dout[n, :, dout_hi, dout_wi].reshape(-1,1,1,1) # (2,1,1,1)
            dx_pad_slice = np.sum(w*n_dout_vals, axis=0) # (3,3,3)
            dx_pad[n, :, dout_hi:dout_hi+HH, dout_wi:dout_wi+WW] += dx_pad_slice
        
        # dw
        # img_comp: part of xpad in forward pass that overlapped, or contributed to,
        # this dout scalar for some example or filter (padding, stride replicated)
        hi, wi = dout_hi*stride, dout_wi*stride
        img_comp = xpad[:, :, hi:hi+HH, wi:wi+WW] # (4,3,3,3)
        for f in range(F):
            # f_dout_vals: for all data points, one filter's dout scalar output
            f_dout_vals = dout[:, f, dout_hi, dout_wi].reshape(-1,1,1,1) # (4,1,1,1)
            dw_slice = np.sum(img_comp*f_dout_vals, axis=0) # (3,3,3)
            dw[f, :, :, :] += dw_slice
            
            # recall forward pass for derivative explanations:
            # img_comp = img[:, hi:hi+HH, wi:wi+WW] # (C,HH,WW) (3,4,4)
            # out[n, f, height, width] = np.sum(img_comp*filter) + f_bias
            
  dx = dx_pad[:, :, pad:-pad, pad:-pad] # (4,3,5,5)
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  Ph = pool_param['pool_height'] # 2
  Pw = pool_param['pool_width'] # 2
  S = pool_param['stride'] # 2
  N, C, H, W = x.shape # (2,3,4,4)

  output_h = int(1 + (H - Ph) / S) # 2
  output_w = int(1 + (W - Pw) / S) # 2
    
  out = np.zeros((N, C, output_h, output_w)) # (2,3,2,2)
  max_idxs = np.zeros(out.shape, dtype=np.ndarray)
  for n in range(N):
    img = x[n]
    for c in range(C):
        img_ch = img[c]
        for h in range(output_h):
            for w in range(output_w):
                hi, wi = h*S, w*S
                pool = img_ch[hi:hi+Ph, wi:wi+Pw]
                out[n, c, h, w] = np.max(pool)
                
                # For backpropagation
                max_i = np.argmax(pool)
                calc_max_i = np.array([hi,wi]) + np.unravel_index(max_i,(Ph, Pw))
                max_idxs[n, c, h, w] = calc_max_i

  cache = (x, max_idxs, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, max_idxs, pool_param = cache
  pool_h = pool_param['pool_height'] # 2
  pool_w = pool_param['pool_width'] # 2
  stride = pool_param['stride'] # 2
  N, C, _, _ = x.shape # (3,2,8,8)

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  # dout: (3,2,4,4)
  dx = np.zeros(x.shape)
  for n in range(N):
    for c in range(C):
        max_idxs_dout_vals = zip(max_idxs[n, c].flatten(), dout[n, c].flatten())
        for (max_i, dout_val) in max_idxs_dout_vals:
            max_hi, max_wi = max_i
            # deriv wrt to x after max pool operation = 0 or x, so set dx at cached
            # coordinates (max_hi, max_wi) to dout_val. otherwise they remain 0.
            dx[n, c, max_hi, max_wi] = dout_val

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorpora=ted. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var: Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  # spatial batch-normalization: reshape the (N, C, H, W) array as an (N*H*W, C)
  # array and perform batch normalization on this array
  N, C, H, W = x.shape
  x_mod = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)
  bn_out, cache = batchnorm_forward(x_mod, gamma, beta, bn_param)
  out = bn_out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout_mod = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
  bn_dx, dgamma, dbeta = batchnorm_backward(dout_mod, cache)
  dx = bn_dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  return dx, dgamma, dbeta