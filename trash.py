def elnet_tfh(X_tfh, y_t, l1_ratio=None, n_folds=3):
    '''
    Run elastic net using scikit-learn ElasticNetCV.
    NB the parameters are different from glmnet:
    l1_ratio here is (1-alpha) in glmnet, i.e. l1_ratio=0 -> ridge; l1_ratio=1 -> lasso
    alpha here is lambda in glmnet
    '''
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

    n_t, n_f, n_h = X_tfh.shape

    enet = ElasticNetCV(cv=n_folds, random_state=0, l1_ratio=l1_ratio)
    enet.fit(X_tfh.reshape(n_t, -1), y_t)

    return {'c': enet.intercept_,
            'k_fh': enet.coef_.reshape(n_f, n_h),
            'enet': enet}

def elnet_tfh_glmnet(X_tfh, y_t, val_idx=None, alpha=np.logspace(-2, 0, 5, 10)):
    '''
    Run elastic net using glmnet
    '''
    if isinstance(alpha, str):
        if alpha == 'lasso':
            print('Using lasso (alpha = 0.01)')
            alpha = [0.01]
        elif alpha == 'ridge':
            print('Using ridge (alpha = 1.0)')
            alpha = [1.0]

    # wrangle data into 2D
    n_t, n_f, n_h = X_tfh.shape
    X_t_fh = X_tfh.reshape(n_t, n_f*n_h)

    # choose fit and validation sets
    if not val_idx:
        print('No validation set specified; just giving best fit on training data')
        fit_idx = range(n_t)
        val_idx = fit_idx
    else:
        fit_idx = list(set(range(n_t)) - set(val_idx))
        fit_idx.sort()

    # run glmnet
    print('Running glmnet on %s alphas' % len(alpha), end='')
    res = []
    for alph in alpha:
        print('.', end='')
        r = glmnet(x=X_t_fh[fit_idx, :], y=y_t[fit_idx],
                   family='gaussian',
                   alpha=alph, standardize=True)
        r['alpha'] = np.zeros((len(r['lambdau']))) + alph
        res.append(r)
    print(' done')

    # paste together the results from runs with multiple alphas
    result = res[0].copy()
    del result['npasses']
    del result['jerr']
    del result['offset']
    del result['class']
    for r in res[1:]:
        for key, value in r.items():
            if key not in result:
                continue
            if len(result[key].shape) == 1:
                axis = 0
            else:
                axis = 1
            result[key] = np.concatenate((result[key], value), axis=axis)

    # fix spelling of lambda
    result['lambda'] = result['lambdau']
    del result['lambdau']

    # get model fits and MSE for each alpha, lambda pair
    y_hat = result['alpha'] + np.dot(X_t_fh[val_idx], result['beta'])
    err = y_hat - y_t[val_idx, None]
    mse = np.sum(err**2, axis=0)/len(y_hat)
    if fit_idx == val_idx:
        print('Choosing hyperparameters based on training data')
        result['fit_mse'] = mse
    else:
        print('Choosing hyperparameters based on validation data')
        result['val_mse'] = mse
    min_idx = np.where(mse == np.amin(mse))[0][0]

    # return best kernel
    kernel = {'c': result['a0'][min_idx],
              'k_fh': result['beta'][:, min_idx].reshape(n_f, n_h)}

    return kernel, result

def tensorize_fht(X_ft, n_h):
    '''
    Probably not to be used -- use _tfh instead
    '''
    n_f, n_t = X_ft.shape
    X_fts = []
    for h in range(n_h-1, -1, -1):
        X_fts.append(np.concatenate([np.zeros((n_f, h))[:, None, :], X_ft[:, None, :n_t-h]], 2))
    return np.concatenate(X_fts, 1)

def tensorize_segments_fht(X_ft, segment_lengths, n_h):
    '''
    Tensorise segments of a seemingly continuous X_ft which is actually composed
    of separate segments. Use _tfh instead.
    '''
    start = 0
    X_fhts = []
    for ln in segment_lengths:
        this_X_ft = X_ft[:, start:start+ln]
        tens = tensorize_fht(this_X_ft, n_h)
        X_fhts.append(tens)
        start = start+ln

    return np.concatenate(X_fhts, 2)

def tensorize_segments_tfh(X_tf, segment_lengths, n_h):
    '''
    Tensorise segments of a seemingly continuous X_ft which is actually composed
    of separate segments.
    '''
    start = 0
    X_tfhs = []
    for ln in segment_lengths:
        this_X_tf = X_tf[start:start+ln, :]
        X_tfhs.append(tensorize_tfh(this_X_tf, n_h))
        start = start+ln
    return np.concatenate(X_tfhs, 0)

class ElNetTFH(ElasticNetCV):
    '''
    Scikit-learn compatible elnet_tfh kernel
    '''
    def __init__(self, *, l1_ratio=None, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=None, positive=False, random_state=None,
                 selection='cyclic'):

        self.kernel = None

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
        n_t, n_f, n_h = X.shape
        super().fit(X.reshape(n_t, -1), y)
        self.kernel = {'c': self.intercept_,
                       'k_fh': self.coef_.reshape(n_f, n_h)
                      }

    def predict(self, X=None):
        n_t = X.shape[0]
        return super().predict(X.reshape(n_t, -1))

    def show(self):
        '''
        Show the kernel
        '''
        show_strf(self.kernel['k_fh'])

class ElNet_TFH_Sigmoid(RegressorMixin):
    def __init__(self, l1_ratio=None):
        if not l1_ratio:
            l1_ratio = [0.001, .25, .5, 0.75, 1]
        self.elnet = ElNetTFH(l1_ratio=l1_ratio)
        self.sigmoid = Sigmoid()

    def fit(self, X=None, y=None):
        self.elnet.fit(X, y)
        pred = self.elnet.predict(X)
        self.sigmoid.fit(pred, y)

    def predict(self, X=None):
        return self.sigmoid.predict(self.elnet.predict(X))

