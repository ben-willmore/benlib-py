import numpy as np

import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef
from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict

def select_folds(n_samples, n_folds):
    fold_len = round(n_samples/n_folds)
    fold_id = np.hstack([[i]*fold_len for i in range(n_folds)])
    fold_id = np.hstack((fold_id, [n_folds-1]*(n_samples-len(fold_id))))
    return [(np.where(fold_id!=i)[0], np.where(fold_id==i)[0]) for i in range(n_folds)]

class CVGLMnet():

    def __init__(self, family='gaussian', alpha=1,
                 n_folds=10, sequential_folds=True,
                 parallel=True, refit_with_best_lambda=True):

        self.family = family
        if alpha=='ridge':
            self.alpha = 0.01
        elif alpha=='lasso':
            self.alpha = 1
        else:
            self.alpha = alpha
        self.n_folds = n_folds
        self.parallel = self.n_folds
        if not parallel:
            self.parallel = 1
        self.sequential_folds = sequential_folds
        self.refit_with_best_lambda = refit_with_best_lambda
        self.cv_result = None
        self.result = None
        self.kernel = None

        if isinstance(family, dict):
        	self.load(family)

    def fit(self, X, y, fold_ids=None):
        # fold_ids (optional) is a vector, same size as y, containing integers 1..n_folds
        # indicating the fold that each data point is in, e.g. [1,1,1,2,2,2,3,3,3]
        # This overrides n_folds and sequential_folds if provided.
        common_params = {'x': X, 'y': y, 'family': self.family, 'alpha': self.alpha}

        cv_params = {'parallel': self.parallel}

        self.cv_result = None

        if fold_ids is not None:
            # use provided fold_ids, don't assign folds
            print('Using user-provided folds for cross-validation')
            cv_params['foldid'] = np.array(fold_ids).astype(int)

        elif self.sequential_folds:
            print('Using %d sequential folds for cross-validation' % self.n_folds)
            n_samples = X.shape[0]
            fold_len = round(n_samples/self.n_folds)
            fold_ids = np.hstack([[i]*fold_len for i in range(self.n_folds)])
            fold_ids = np.hstack((fold_ids, [self.n_folds-1]*(n_samples-len(fold_ids)))).astype(int)
            cv_params['foldid'] = fold_ids

        else:
            print('Using %d random folds for cross-validation' % self.n_folds)
            cv_params['nfolds'] = self.n_folds

        self.result = cvglmnet(**common_params, **cv_params)

        if self.refit_with_best_lambda:
            print('Refitting with best lambda on whole training set')
            self.cv_result = self.result
            self.result = glmnet(**common_params,
                                 lambdau=self.cv_result['lambda_min'])
            coeffs = glmnetCoef(self.result, s=self.result['lambdau'])[:,0]
        else:
            coeffs = cvglmnetCoef(self.result, s=self.result['lambda_min'])
        self.kernel = {'a': coeffs[0],
                       'beta': coeffs[1:]}

    def predict(self, X):
        if self.result['class'] == 'elnet':
            return np.squeeze(glmnetPredict(self.result, X))
        else:
            return np.squeeze(cvglmnetPredict(self.result, X))

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.corrcoef(y, y_hat)[0,1]

    def dump(self):
        result = self.result.copy()
        del result['nulldev']
        info = self.__dict__.copy()
        info['result'] = result
        return info

    def load(self, dct):
        for key, val in dct.items():
            setattr(self, key, val)

class CVGLMnet_tfh(CVGLMnet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_t = None
        self.n_f = None
        self.n_h = None

    def fit(self, X, y, *args, **kwargs):
        self.n_t, self.n_f, self.n_h = X.shape
        X = X.reshape(self.n_t, self.n_f*self.n_h)
        super().fit(X, y, *args, **kwargs)
        self.kernel = {'c': self.kernel['a'],
            'k_fh': self.kernel['beta'].reshape(self.n_f, self.n_h)}

    def predict(self, X):
        X = X.reshape(X.shape[0], self.n_f*self.n_h)
        return super().predict(X)

    def score(self, X, y):
        X = X.reshape(X.shape[0], self.n_f*self.n_h)
        return super().score(X, y)
