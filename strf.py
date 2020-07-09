'''
STRF estimation and display functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
#from glmnet_python import glmnet, glmnetSet
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNetCV
from scipy.linalg import lstsq

def show_strf(k_fh):
    '''
    Show STRF using blue->red colormap, scaled so that zero is in the centre (white)
    '''
    mx = max([np.max(k_fh), np.abs(np.min(k_fh))])
    plt.imshow(k_fh, cmap='bwr', vmin=-mx, vmax=mx)

def tensorize_segments(X, n_h):
    '''
    If X is a list, tensorize segments and concatenate them.
    If X is 3D, do nothing
    If X is 2D, tensorize it
    '''
    if isinstance(X, list):
        X = [tensorize_tfh(x, n_h) for x in X]
        return np.concatenate(X, 0)
    if len(X.shape) == 3:
        return X
    return tensorize_tfh(X, n_h)

def concatenate_segments(y):
    '''
    If y is a list, concatenate the segments, otherwise leave it alone
    '''
    if isinstance(y, list):
        return np.concatenate(y, 0)
    return y

def tensorize_tfh(X_tf, n_h):
    '''
    Tensorise a continuous X_tf, to produce X_tfh with n_h history steps
    '''
    n_t, n_f = X_tf.shape
    X_tfs = []
    for h in range(n_h-1, -1, -1):
        X_tfs.append(np.concatenate([np.zeros((h, n_f)), X_tf[:n_t-h, :]], 0))
    return np.stack(X_tfs, axis=-1)

def split_tx(X_tf, segment_lengths):
    '''
    Split up a seemingly continuous X_tf into a list of its contiguous segments
    '''
    n_t = X_tf.shape[0]
    segment_idx = np.cumsum(segment_lengths)
    assert segment_idx[-1] == n_t
    segment_idx = segment_idx[:-1]
    return np.split(X_tf, segment_idx, axis=0)

def reconstruct_gaps_tx(X_tx, segment_lengths, n_h):
    '''
    Reintroduce n_h gaps into an X_tx matrix, to produce a continuous time
    series that doensn't have the wrong history at the beginning of
    each segment
    '''
    shp = X_tx.shape
    n_t = shp[0]
    X_tf = X_tx.reshape(n_t, -1)

    n_t, n_f = X_tf.shape
    segment_idx = np.cumsum(segment_lengths)
    assert segment_idx[-1] == n_t
    segment_idx = segment_idx[:-1]
    X_tfs = np.split(X_tf, segment_idx, axis=0)
    X_tfs = [np.concatenate((np.zeros((n_h-1, n_f)), x), 0) for x in X_tfs]
    X_tf = np.concatenate(X_tfs, 0)

    if len(shp) == 1:
        return X_tf.reshape([X_tf.shape[0]])

    return X_tf.reshape([X_tf.shape[0]].extend(shp[1:]))

class ElNet(ElasticNetCV):
    '''
    Scikit-learn compatible elnet kernel. Works with either a continuous X_tf,
    or a list of X_tfs.
    '''
    def __init__(self, n_h=15, *, l1_ratio=None, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=None, positive=False, random_state=None,
                 selection='cyclic'):

        self.kernel = None
        self.n_h = n_h

        if isinstance(l1_ratio, str):
            if l1_ratio == 'lasso':
                print('Using lasso (l1_ratio = 1.0)')
                l1_ratio = 1.0
            elif l1_ratio == 'ridge':
                print('Using ridge (l1_ratio = 0.001)')
                l1_ratio = 0.001
        elif not l1_ratio:
            l1_ratio = [0.001, .25, .5, 0.75, 1]
            print('Using a range of l1_ratios: %s' % str(l1_ratio))

        super().__init__(l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas, alphas=alphas,
                         fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                         max_iter=max_iter, tol=tol, cv=cv, copy_X=copy_X,
                         verbose=verbose, n_jobs=n_jobs, positive=positive,
                         random_state=random_state, selection=selection)

    def fit(self, X=None, y=None):
        X = tensorize_segments(X, self.n_h)
        y = concatenate_segments(y)
        n_t, n_f, n_h = X.shape
        n_t, n_f, n_h = X.shape
        super().fit(X.reshape(n_t, -1), y)
        self.kernel = {'c': self.intercept_,
                       'k_fh': self.coef_.reshape(n_f, n_h)
                      }

    def predict(self, X=None):
        X = tensorize_segments(X, self.n_h)
        n_t = X.shape[0]
        return super().predict(X.reshape(n_t, -1))

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])


class SeparableKernel(RegressorMixin):
    '''
    Scikit-learn compatible separable kernel
    '''

    def __init__(self, n_h=15, n_iter=15):
        self.n_iter = n_iter
        self.kernel = None
        self.n_h = n_h

    def fit(self, X=None, y=None):
        X = tensorize_segments(X, self.n_h)
        y = concatenate_segments(y)
        self.kernel = self.sepkernel_tfh(X, y, n_iter=self.n_iter)

    def predict(self, X=None):
        X = tensorize_segments(X, self.n_h)
        return self.sepconv_tfh(X, self.kernel)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def sepconv_tfh(self, X_tfh, kernel):
        a_th = np.tensordot(X_tfh, kernel['k_f'], axes=(1, 0))
        y_t = kernel['c'] + np.tensordot(a_th, kernel['k_h'], axes=(1, 0))
        return y_t

    def sepkernel_tfh(self, X_tfh, y_t, n_iter=15):
        '''
        Estimate separable kernel
        '''
        n_t, n_f, n_h = X_tfh.shape

        # subtract off means
        X_mn = np.mean(X_tfh, axis=0)
        X_tfh = X_tfh - X_mn

        y_mn = np.mean(y_t)
        y_t = y_t - y_mn

        # estimate k_f and k_h
        k_f = np.ones(n_f)
        k_h = np.ones(n_h)

        for _ in range(n_iter):
            yh = np.tensordot(X_tfh, k_f, axes=(1, 0))
            # yh = np.sum(np.multiply(X_tfh, k_f[None,:,None]), 1)
            k_h = lstsq(yh, y_t)[0]

            yf = np.tensordot(X_tfh, k_h, axes=(2, 0))
            # yf = np.sum(np.multiply(X_tfh, k_h[None,None,:]), 2)
            k_f = lstsq(yf, y_t)[0]

        # we need to convert the kernel back into un-normalised space
        # this is adapted from blasso.m -- where both mean and SD of coefficients are adjusted
        # before fitting. Here, only the mean is altered.

        # we have fit the equation of a line where y' = m'_i x'_i + k' (actually k' = 0)
        # and y' = y-mu_y and x'_i = x_i - mu_xi
        # by substitution and simplification, we can get y = m_i x_i + k
        # where,
        # kernel coefficients m_i = m'_i (unchanged)
        # offset k = mu_y + Sum(m'_i * mu_xi)

        k_fh = np.outer(k_f, k_h)
        kernel = {'c': y_mn - np.sum(np.multiply(k_fh, X_mn)),
                  'k_fh': k_fh,
                  'k_f': k_f,
                  'k_h': k_h
                 }

        return kernel

