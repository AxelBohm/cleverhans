"""The Wasserstein robust L2 penalty formulation """


import numpy as np
import tensorflow as tf

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.utils import compute_gradient


def wrm(model_fn, x, x_true, eps, nb_iter, loss_fn=None, y=None, targeted=False):
    """
    This function implements the Lagrange penalty formulation of the 
    distriutionally robust Wasserstein method described in 
    https://arxiv.org/pdf/1710.10571.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: .5 / gamma (Lagrange dual parameter) 
    :param nb_iter: Number of attack iterations.
    :param loss_fn: (optional) callable. loss function that takes (labels, logits) as arguments and returns loss.
                    default function is 'tf.nn.sparse_softmax_cross_entropy_with_logits'
    :param y: (optional) Tensor with true labels.  
    :return: a tensor for the adversarial example
    """

    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)

    x_adv = x

    for t in range(nb_iter):
        # loss = loss_fn(y, model_fn(x_adv))
        # grad, = tf.gradients(eps*loss, x_adv)
        grad = compute_gradient(model_fn, loss_fn, x_adv, y, targeted)

        grad2 = x_adv-x_true
        grad = eps*grad - grad2
        x_adv = x_adv+1./np.sqrt(t+1)*grad

    return x_adv
