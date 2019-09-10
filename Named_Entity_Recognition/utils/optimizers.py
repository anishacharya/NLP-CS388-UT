# optimizers.py

from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from typing import List


class Optimizer(ABC):
    """
    Optimizer that aims to *maximize* a given function.
    """

    def score(self, feats: List[int]):
        """
        :param feats: List[int] feature vector indices (i.e., sparse representation of a feature vector)
        :return: floating-point score
        """
        i = 0
        score = 0.0
        while i < len(feats):
            score += self.access(feats[i])
            i += 1
        return score

    @abstractmethod
    def apply_gradient_update(self, gradient: Counter, batch_size: int):
        pass

    @abstractmethod
    def access(self, i: int):
        pass

    @abstractmethod
    def get_final_weights(self):
        pass


#
class SGDOptimizer(Optimizer):
    """
    SGD optimizer implementation, designed to have the same interface as the Adagrad optimizers

    Attributes:
        weights: numpy array containing initial settings of the weights. Usually initialize to the 0 vector unless
        you have a very good reason not to.
        alpha: step size
    """
    def __init__(self, init_weights: np.ndarray, alpha):
        self.weights = init_weights
        self.alpha = alpha

    def apply_gradient_update(self, gradient: Counter, batch_size: int):
        """
        Take a sparse representation of the gradient and make an update, normalizing by the batch size to keep
        hyperparameters constant as the batch size is varied
        :param gradient: Counter containing the gradient values (i.e., sparse representation of the gradient)
        :param batch_size: how many examples the gradient was computed on
        :return: nothing, modifies weights in-place
        """
        for i in gradient.keys():
            self.weights[i] = self.weights[i] + self.alpha * gradient[i]

    def access(self, i: int):
        """
        :param i: index of the weight to access
        :return: value of that weight
        """
        return self.weights[i]

    def get_final_weights(self):
        return self.weights


#
class L1RegularizedAdagradTrainer(Optimizer):
    """
    Wraps a weight vector and applies the Adagrad update using second moments of features to make custom step sizes.
    This version incorporates L1 regularization: while this regularization should be applied to squash the feature vector
    on every gradient update, we instead evaluate the regularizer lazily only when the particular feature is touched
    (either by gradient update or by access). approximate lets you turn this off for faster access, but regularization is
    now applied somewhat inconsistently.
    See section 5.1 of http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf for more details
    """

    def __init__(self, init_weights, lamb=1e-8, eta=1.0, use_regularization=False, approximate=True):
        """
        :param init_weights: a numpy array of the correct dimension, usually initialized to 0
        :param lamb: float lambda constant for the regularizer. Values above 0.01 will often cause all features to be zeroed out.
        :param eta: float step size. Values from 0.01 to 10 often work well.
        :param use_regularization:
        :param approximate: turns off gradient updates on access, only uses them when weights are written to.
        So regularization is applied inconsistently, but it makes things faster.
        """
        self.weights = init_weights
        self.lamb = lamb
        self.eta = eta
        self.use_regularization = use_regularization
        self.approximate = approximate
        self.curr_iter = 0
        self.last_iter_touched = [0 for i in range(0, self.weights.shape[0])]
        self.diag_Gt = np.zeros_like(self.weights, dtype=float)

    def apply_gradient_update(self, gradient: Counter, batch_size: int):
        """
        Take a sparse representation of the gradient and make an update, normalizing by the batch size to keep
        hyperparameters constant as the batch size is varied
        :param gradient Counter containing the gradient values (i.e., sparse representation of the gradient)
        :param batch_size: how many examples the gradient was computed on
        :return: nothing, modifies weights in-place
        """
        batch_size_multiplier = 1.0 / batch_size
        self.curr_iter += 1
        for i in gradient.keys():
            xti = self.weights[i]
            # N.B. We negate the gradient here because the Adagrad formulas are all for minimizing
            # and we're trying to maximize, so think of it as minimizing the negative of the objective
            # which has the opposite gradient
            # See section 5.1 in http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf for more details
            # eta is the step size, lambda is the regularization
            gti = -gradient[i] * batch_size_multiplier
            old_eta_over_Htii = self.eta / (1 + np.sqrt(self.diag_Gt[i]))
            self.diag_Gt[i] += gti * gti
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            new_xti = xti - eta_over_Htii * gti
            # Apply the regularizer for every iteration since touched
            iters_since_touched = self.curr_iter - self.last_iter_touched[i]
            self.last_iter_touched[i] = self.curr_iter
            self.weights[i] = np.sign(new_xti) * max(0, np.abs(new_xti) - self.lamb * eta_over_Htii - (iters_since_touched - 1) * self.lamb * old_eta_over_Htii)

    def access(self, i: int):
        """
        :param i: index of the weight to access
        :return: value of that weight
        """
        if not self.approximate and self.last_iter_touched[i] != self.curr_iter:
            xti = self.weights[i]
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            iters_since_touched = self.curr_iter - self.last_iter_touched[i]
            self.last_iter_touched[i] = self.curr_iter
            self.weights[i] = np.sign(xti) * max(0, np.abs(xti) - iters_since_touched * self.lamb * self.eta * eta_over_Htii);
        return self.weights[i]

    def get_final_weights(self):
        """
        :return: a numpy array containing the final weight vector values -- manually calls access to force each weight to
        have an up-to-date value.
        """
        for i in range(0, self.weights.shape[0]):
            self.access(i)
        return self.weights


class UnregularizedAdagradTrainer(Optimizer):
    """
    Applies the Adagrad update with no regularization. Will be substantially faster than the L1 regularized version
    due to less computation required to update each feature. Same interface as the regularized version.
    """
    def __init__(self, init_weights, eta=1.0):
        self.weights = init_weights
        self.eta = eta
        self.diag_Gt = np.zeros_like(self.weights, dtype=float)

    def apply_gradient_update(self, gradient: Counter, batch_size: int):
        batch_size_multiplier = 1.0 / batch_size
        for i in gradient.keys():
            xti = self.weights[i]
            gti = -gradient[i] * batch_size_multiplier
            self.diag_Gt[i] += gti * gti
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            self.weights[i] = xti - eta_over_Htii * gti

    def access(self, i: int):
        return self.weights[i]

    def get_final_weights(self):
        return self.weights
