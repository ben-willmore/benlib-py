'''
STRF estimation and display functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from scipy.linalg import lstsq
from benlib.utils import calc_CC_norm

def show_strf(k_fh, xlim=None, ylim=None, xticks=None, yticks=None, sort_order=None):
    '''
    Show STRF using blue->red colormap, scaled so that zero is in the centre (white)
    '''
    if sort_order is not None:
        k_fh = np.array(k_fh)[sort_order, :]

    mx = max([np.max(k_fh), np.abs(np.min(k_fh))])

    if xlim is None:
        xlim = [0, k_fh.shape[1]-1]
    xvals = np.linspace(np.min(xlim), np.max(xlim), k_fh.shape[1])

    if ylim is None:
        ylim = [0, k_fh.shape[0]-1]
    yvals = np.linspace(np.min(ylim), np.max(ylim), k_fh.shape[0])

    plt.pcolormesh(xvals, yvals, k_fh, cmap='seismic', vmin=-mx, vmax=mx)

    ax = plt.gca()

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)


    # # if aspect ratio is too extreme, scale so the plot is square
    # ratio = (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
    # if ratio > 4:
    #     ax.set_aspect(1/ratio)

def get_bf(k_fh):
    '''
    Get best frequency from k_fh -- defined here as the frequency
    with highest positive coefficient summed over time. Negative
    coefficients are ignored.
    '''
    sum_pos = np.maximum(k_fh, 0).sum(axis=1)

    idx_max = np.where(sum_pos==sum_pos.max())[0]
    if idx_max.shape[0]>1:
        return int(np.median(idx_max))
    return idx_max[0]

def select_idxes(X_tfs, idxes):
    '''
    Choose indices from list of X_tfs (much like X_tf[idxes, :] but for a list of
    X_tfs. Suprisingly annoying / difficult to write
    '''
    if len(X_tfs[0].shape) == 1:
        one_d = True
        X_tfs = [x.reshape((x.shape[0], 1)) for x in X_tfs]
    else:
        one_d = False

    selected_idxes = list(set(idxes))
    selected_idxes.sort()

    segment_lengths = [0] + [len(seg) for seg in X_tfs]
    if np.max(idxes) >= np.sum(segment_lengths):
        raise ValueError('Largest idx >= total data length')
    segment_starts = np.cumsum(segment_lengths)[:-1]

    X_tfs_selected = []

    for seg_idx, segment in enumerate(X_tfs):
        start = None
        for rel_idx in range(segment.shape[0]):
            overall_idx = segment_starts[seg_idx] + rel_idx
            if overall_idx in selected_idxes:
                if start is None:
                    start = rel_idx
            else:
                if start is not None:
                    X_tfs_selected.append(segment[start:rel_idx, :])
                    start = None
        if start is not None:
            X_tfs_selected.append(segment[start:, :])

    if one_d:
        X_tfs_selected = [x.ravel() for x in X_tfs_selected]

    return X_tfs_selected

def test_select_idxes(X_tfs=None, idxes=None):
    '''
    Test select_idxes
    '''
    if X_tfs is None:
        sz = [(np.random.randint(1, 100), 10) for x in range(np.random.randint(1, 5))]
    else:
        sz = [np.shape(x) for x in X_tfs]

    X_tfs = []
    # assign segment_number + small random number to each element
    for i, sh in enumerate(sz):
        X_tfs.append(i * np.ones(sh) + np.random.random(sh)/3)

    # check that floor() of value is equal to segment number in all cases
    for x in X_tfs:
        assert np.all(np.floor(x) == np.floor(x[0, 0]))
    vals = [int(np.floor(x[0, 0])) for x in X_tfs]
    assert all([v == i for v, i in enumerate(vals)])

    if idxes is None:
        l = [x.shape[0] for x in X_tfs]
        n_t = sum(l)
        idxes = np.random.permutation(n_t)[:int(np.ceil(n_t/2))]

    X_tfs_selected = select_idxes(X_tfs, idxes)
    X_tf_selected = np.concatenate(X_tfs_selected, 0)

    idxes = list(set(idxes))
    idxes.sort()

    X_tf_check = np.concatenate(X_tfs, 0)
    X_tf_check = X_tf_check[idxes, :]

    # check that the right number of indices have been selected
    assert X_tf_selected.shape[0] == len(idxes)

    # check they have the right values (including the random part)
    assert np.array_equal(X_tf_selected, X_tf_check)

    # check that every segment contains only values from a single original
    # segment, i.e. no segments have got combined
    for x in X_tfs_selected:
        assert np.all(np.floor(x) == np.floor(x[0]))
    return True

def combine_segments(X1, X2):
    '''
    Combine two lists of 1d vectors (e.g. containing x_t and z_t) into
    one list
    '''
    return [np.stack((x1, x2), axis=1) for x1, x2 in zip(X1, X2)]

def tensorize_segments(X, n_h, n_fut=0):
    '''
    If X is a list, tensorize segments and concatenate them.
    If X is 3D, do nothing
    If X is 2D, tensorize it
    '''
    if isinstance(X, list):
        X = [tensorize_tfh(x, n_h, n_fut) for x in X]
        return np.concatenate(X, 0)
    if len(X.shape) == 3:
        return X
    return tensorize_tfh(X, n_h, n_fut)

def concatenate_segments(y, mean=False):
    '''
    If y is a list, concatenate the segments, otherwise leave it alone
    '''
    if isinstance(y, list):
        y = np.concatenate(y, 0)
    if mean and len(y.shape) == 2:
        y = np.mean(y, 1)
    return y

def tensorize_tfh(X_tf, n_h, n_fut=0):
    '''
    Tensorise a continuous X_tf, to produce X_tfh with n_h history steps
    and optionally n_fut future steps
    '''
    n_t, n_f = X_tf.shape
    X_tfs = []
    for h in range(n_h-1, -1, -1):
        X_tfs.append(np.concatenate([np.zeros((h, n_f)), X_tf[:n_t-h, :]], 0))
    for h in range(1, n_fut+1):
        X_tfs.append(np.concatenate([X_tf[h:, :], np.zeros((h, n_f))], 0))

    X_tfh = np.stack(X_tfs, axis=-1)
    return X_tfh

def test_tensorize_tfh():
    '''
    Test tensorize_tfh using arandom sized array
    '''
    n_t, n_f = np.random.randint(1, 50, (2))
    n_h, n_fut = np.random.randint(1, n_t+1, (2))
    print(n_t, n_f, n_h, n_fut)
    X_tf = np.tile(np.arange(n_t)[:, np.newaxis], (1, n_f))
    X_fht = tensorize_tfh(X_tf, n_h=n_h, n_fut=n_fut)
    print(X_fht.shape)

    success = True
    for t in range(n_t):
        for f in range(n_f):
            for h in range(0, n_h+n_fut):
                # the zero-time bin should be in the last history slot,
                # so we expect x=t when h=(n_h-1)
                # so offset should be zero when h=n_h-1
                x = max(t+h-n_h+1, 0)
                if x > n_t-1:
                    x = 0
                if x != X_fht[t, f, h]:
                    success = False
                    print('(%d, %d, %d): expected %d, actual %d' % (t, f, h, x, X_fht[t, f, h]))
    return success

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
    Reintroduce n_h-1 gaps into an X_tx matrix, to produce a continuous time
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
    or a list of X_tfs. To make this faster, you can use 3 folds (cv=3),
    reduce n_alphas (to say 10), and set l1_ratio to 'lasso' or 'ridge'
    '''
    def __init__(self, l1_ratio=None, eps=1e-3,
                 n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=-1, positive=False, random_state=None,
                 selection='cyclic'):

        self.kernel = None
        self.l1_ratio = None

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
        '''
        Fit elnet model
        '''
        n_t, n_f, n_h = X.shape

        super().fit(X.reshape(n_t, -1), y)
        self.kernel = {'type': 'ElNet',
                       'n_f': n_f,
                       'n_h': n_h,
                       'c': self.intercept_,
                       'k_fh': self.coef_.reshape(n_f, n_h),
                       'alpha': self.alpha_,
                       'l1_ratio': self.l1_ratio_
                      }

    def predict(self, X=None):
        '''
        Predictions of elnet model
        '''
        n_t = X.shape[0]
        return super().predict(X.reshape(n_t, -1))

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score of elnet model
        '''
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel

class ElNetNoCV(ElasticNet):
    '''
    ** VERY ROUGH **
    Scikit-learn compatible elnet kernel. No hyperparameter selection through cross-validation.
    '''
    def __init__(self,
                 alpha=1.0, l1_ratio=1.0,
                 fit_intercept=True, normalize=False, precompute=False,
                 max_iter=1000, tol=1e-4, warm_start=False,
                 positive=False, random_state=None,
                 selection='cyclic'):

        if isinstance(l1_ratio, str):
            if l1_ratio == 'lasso':
                print('Using lasso (l1_ratio = 1.0)')
                l1_ratio = 1.0
            elif l1_ratio == 'ridge':
                print('Using ridge (l1_ratio = 0.001)')
                l1_ratio = 0.001

        self.kernel = None

        super().__init__(alpha=alpha, l1_ratio=l1_ratio,
                         fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                         max_iter=max_iter, tol=tol, warm_start=warm_start,
                         positive=positive,
                         random_state=random_state, selection=selection)

    def fit(self, X=None, y=None):
        '''
        Fit elnet model
        '''
        n_t, n_f, n_h = X.shape

        super().fit(X.reshape(n_t, -1), y)

        self.kernel = {'type': 'ElNetNoCV',
                       'n_f': n_f,
                       'n_h': n_h,
                       'c': self.intercept_,
                       'k_fh': self.coef_.reshape(n_f, n_h),
                       'alpha': self.alpha,
                       'l1_ratio': self.l1_ratio
                      }

    def predict(self, X=None):
        '''
        Predictions of elnet model
        '''
        n_t = X.shape[0]

        return super().predict(X.reshape(n_t, -1))

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score of elnet model
        '''
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel


class ElNetLassoSubset():
    '''
    Experimental faster elnet using small groups of regressors to select promising ones (those
    that have coefficients > 0), then doing final full regression using elnet
    only on the selected regressors
    '''

    def __init__(self, l1_ratio='lasso',
                 group_size=50, max_regressors=None, eps=1e-3,
                 n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=-1, positive=False, random_state=None,
                 selection='cyclic'):

        self.group_size = group_size
        self.max_regressors = max_regressors

        self.subset_model = ElNet(l1_ratio='lasso', eps=eps,
                 n_alphas=n_alphas, alphas=alphas,
                 fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                 max_iter=max_iter, tol=tol, cv=cv, copy_X=copy_X,
                 verbose=verbose, n_jobs=n_jobs, positive=positive, random_state=random_state,
                 selection=selection)
        self.model = ElNet(l1_ratio=l1_ratio, eps=eps,
                 n_alphas=n_alphas, alphas=alphas,
                 fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                 max_iter=max_iter, tol=tol, cv=cv, copy_X=copy_X,
                 verbose=verbose, n_jobs=n_jobs, positive=positive, random_state=random_state,
                 selection=selection)
        self.n_regressors = None
        self.included_regressors = None
        self.kernel = None

    def fit(self, X=None, y=None):
        self.n_regressors = X.shape[1]
        coeff = np.zeros((self.n_regressors))
        for start in np.arange(0, self.n_regressors, self.group_size):
            subset = X[:,start:start+self.group_size,:]
            self.subset_model.fit(subset, y)
            k_fh = self.subset_model.dump()['k_fh']
            coeff[start:start+subset.shape[1]] = np.sum(np.square(k_fh), axis=1)

        # get regressors with non-zero coefficients
        self.included_regressors = np.argwhere(coeff>0)[:,0]

        # select just one regressor if all coeffs are zero
        if len(self.included_regressors) == 0:
            self.included_regressors = np.array([0])

        if self.max_regressors is not None:
            # include only the top n regressors
            if len(self.included_regressors) > self.max_regressors:
                included_coeff = coeff[self.included_regressors]
                order = np.argsort(-included_coeff)

                self.included_regressors = np.sort(self.included_regressors[order[:self.max_regressors]])

        subset_tfh = X[:,self.included_regressors,:]

        self.model.fit(subset_tfh, y)
        k_fh_subset = self.model.coef_.reshape(self.model.kernel['n_f'], self.model.kernel['n_h'])

        n_h = X.shape[2]
        k_fh = np.zeros((self.n_regressors, n_h))
        k_fh[self.included_regressors,:] = k_fh_subset

        self.kernel = {'type': 'ElNetLassoSubset',
                       'n_f': self.n_regressors,
                       'n_h': n_h,
                       'c': self.model.intercept_,
                       'k_fh': k_fh,
                       'k_fh_subset': k_fh_subset,
                       'included_regressors': self.included_regressors,
                       'alpha': self.model.alpha_,
                       'l1_ratio': self.model.l1_ratio_
                      }

    def predict(self, X=None):
        return self.model.predict(X[:,self.included_regressors,:])

    def score(self, X=None, y=None, sample_weight=None):
        subset_tfh = X[:,self.included_regressors,:]
        return self.model.score(subset_tfh, y)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel

class ElNetRidgeSubset():
    '''
    Experimental faster elnet using small groups of regressors to select promising ones (in order of
    decreasing coefficient values using ridge regression), then doing final full regression using
    elnet only on the selected regressors
    '''

    def __init__(self, l1_ratio='lasso',
                 n_regressors_to_include=100, group_size=50, eps=1e-3,
                 n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=-1, positive=False, random_state=None,
                 selection='cyclic'):

        self.n_regressors_to_include = n_regressors_to_include
        self.group_size = group_size
        self.subset_model = ElNet(l1_ratio='ridge', eps=eps,
                 n_alphas=n_alphas, alphas=alphas,
                 fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                 max_iter=max_iter, tol=tol, cv=cv, copy_X=copy_X,
                 verbose=verbose, n_jobs=n_jobs, positive=positive, random_state=random_state,
                 selection=selection)
        self.model = ElNet(l1_ratio=l1_ratio, eps=eps,
                 n_alphas=n_alphas, alphas=alphas,
                 fit_intercept=fit_intercept, normalize=normalize, precompute=precompute,
                 max_iter=max_iter, tol=tol, cv=cv, copy_X=copy_X,
                 verbose=verbose, n_jobs=n_jobs, positive=positive, random_state=random_state,
                 selection=selection)
        self.n_regressors = None
        self.included_regressors = None
        self.kernel = None

    def fit(self, X=None, y=None):
        self.n_regressors = X.shape[1]
        coeff = np.zeros((self.n_regressors))
        for start in np.arange(0, self.n_regressors, self.group_size):
            subset = X[:,start:start+self.group_size,:]
            self.subset_model.fit(subset, y)
            k_fh = self.subset_model.dump()['k_fh']
            coeff[start:start+subset.shape[1]] = np.sum(np.square(k_fh), axis=1)
        order = np.argsort(-coeff)
        self.included_regressors = np.sort(order[:self.n_regressors_to_include])
        subset_tfh = X[:,self.included_regressors,:]

        self.model.fit(subset_tfh, y)
        k_fh_subset = self.model.coef_.reshape(self.model.kernel['n_f'], self.model.kernel['n_h'])

        n_h = X.shape[2]
        k_fh = np.zeros((self.n_regressors, n_h))
        k_fh[self.included_regressors,:] = k_fh_subset

        self.kernel = {'type': 'ElNetRidgeSubset',
                       'n_f': self.n_regressors,
                       'n_h': n_h,
                       'c': self.model.intercept_,
                       'k_fh': k_fh,
                       'k_fh_subset': k_fh_subset,
                       'included_regressors': self.included_regressors,
                       'alpha': self.model.alpha_,
                       'l1_ratio': self.model.l1_ratio_
                      }

    def predict(self, X=None):
        return self.model.predict(X[:,self.included_regressors,:])

    def score(self, X=None, y=None, sample_weight=None):
        subset_tfh = X[:,self.included_regressors,:]
        return self.model.score(subset_tfh, y)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel

class LinearKernel():
    '''
    Shell class for reloading dumped kernels.
    '''
    def __init__(self):
        self.kernel = None

    def reload(self, kernel):
        self.kernel = kernel

    def predict(self, X=None):
        '''
        Predictions of separable kernel model
        '''
        return conv_tfh(X, self.kernel)

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Sore of separable kernel model
        '''
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel

class SeparableKernel(RegressorMixin):
    '''
    Scikit-learn compatible separable kernel
    '''

    def __init__(self, n_iter=15):
        self.n_iter = n_iter
        self.kernel = None

    def fit(self, X=None, y=None):
        '''
        Fit separable kernel model
        '''
        self.kernel = sepkernel_tfh(X, y, n_iter=self.n_iter)
        self.kernel['type'] = 'SeparableKernel'

    def predict(self, X=None):
        '''
        Predictions of separable kernel model
        '''
        return sepconv_tfh(X, self.kernel)

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Sore of separable kernel model
        '''
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel

def conv_tfh(X_tfh, kernel):
    '''
    Response of inseparable kernel
    '''
    if isinstance(kernel, dict):
        y_t = kernel['c'] + np.tensordot(X_tfh, kernel['k_fh'], axes=((1, 2), (0, 1)))
    else:
        y_t = np.tensordot(X_tfh, kernel, axes=((1, 2), (0, 1)))
    return y_t

def sepconv_tfh(X_tfh, kernel):
    '''
    Response of separable kernel
    '''
    a_th = np.tensordot(X_tfh, kernel['k_f'], axes=(1, 0))
    y_t = kernel['c'] + np.tensordot(a_th, kernel['k_h'], axes=(1, 0))
    return y_t

def sepkernel_tfh(X_tfh, y_t, n_iter=15):
    '''
    Estimate separable kernel
    '''
    _, n_f, n_h = X_tfh.shape

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
              'n_f': n_f,
              'n_h': n_h,
              'k_fh': k_fh,
              'k_f': k_f,
              'k_h': k_h
             }

    return kernel

class RankNKernel(RegressorMixin):
    '''
    Scikit-learn compatible rank-n kernel
    '''

    def __init__(self, n_h=15, n_fut=0, n_folds=10, n_iter=15):
        self.n_iter = n_iter
        self.kernel = None
        self.n_folds = n_folds

    def fit(self, X=None, y=None, check=False):
        '''
        Fit rank-n kernel model.
        '''
        n_t, n_f, n_h = X.shape

        kfolds = KFold(n_splits=self.n_folds)
        sepkernel = SeparableKernel(n_iter=self.n_iter)

        resid = y
        rank = 0

        # find best rank using cross-validation
        while True:
            scores = np.zeros(self.n_folds)
            for i, (train_idx, test_idx) in enumerate(kfolds.split(X)):
                sepkernel.fit(X[train_idx, :], resid[train_idx])
                scores[i] = sepkernel.score(X[test_idx, :], resid[test_idx])
            score = np.mean(scores)
            if score > 0:
                resid = resid - sepkernel.predict(X)
                rank = rank + 1
            else:
                break

        # force rank to be at least 1, even if the model just doesn't fit
        rank = max(rank, 1)

        # refit kernel on all data using best rank
        resid = y
        self.kernel = {'type': 'RankNKernel',
                       'n_f': n_f,
                       'n_h': n_h,
                       'c': 0,
                       'k_fh': np.zeros((n_f, n_h)),
                       'rank': rank,
                       }
        for _ in range(rank):
            sepkernel.fit(X, resid)
            self.kernel['c'] = self.kernel['c'] + sepkernel.kernel['c']
            self.kernel['k_fh'] = self.kernel['k_fh'] + sepkernel.kernel['k_fh']
            resid = resid - sepkernel.predict(X)

        # verify that predictions of overall kernel are identical to those
        # of the summed kernels
        if check:
            resid_overall = y - self.predict(X)
            assert np.all(np.abs(resid - resid_overall) < 1e-10)

    def predict(self, X=None):
        '''
        Predictions of separable kernel model
        '''
        return conv_tfh(X, self.kernel)

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Sore of separable kernel model
        '''
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])


    def dump(self):
        '''
        Return most important parameters in a pickleable format
        '''
        return self.kernel

class SplitPct():
    '''
    Cross-validation splitter; single split by training percentage, not shuffled
    '''
    def __init__(self, train_pct=90):
        self.train_pct = train_pct

    def split(self, X=None, y=None, groups=None):
        n = X.shape[0]
        n_test = int(self.train_pct/100 * n)
        return [(np.arange(n_test), np.arange(n_test, n))]
