def melbank3(n_filters, n_fft, f_s, f_lo=0, f_hi=0.5, typ='log'):
    if typ == 'log':
        freq2scale = np.log10
        scale2freq = lambda x: np.power(10, x)
    elif typ == 'erb':
        freq2scale = frq2erb
        scale2freq = erb2frq
    elif typ == 'cat':
        freq2scale = frq2erb_cat
        scale2freq = erb2frq_cat

    # lo and hi filter frequencies in the desired units
    melfreq_lo, melfreq_hi = freq2scale([f_lo * f_s, f_hi * f_s])
    # fixed increment required to get from melfreq_lo to melfreq_hi in n_filters steps
    melinc = (melfreq_hi - melfreq_lo)/(n_filters-1)
    # add 2 extra increments because the filters overlap by a factor of 2
    melfreq_lo, melfreq_hi = melfreq_lo-melinc, melfreq_hi+melinc
    # get fl_lo, fl_mid, fh_mid, fh_hi in Hz
    frq = scale2freq(melfreq_lo + np.array([0, 1, n_filters, n_filters+1]) * melinc)
    blim = frq * n_fft / f_s
    print('blim', blim)
    centre_freqs = melfreq_lo + np.arange(1, n_filters + 1) * melinc
    print(melfreq_lo, melfreq_hi)
    freqs = melfreq_lo + np.arange(0, n_filters+2) * melinc
    print(np.min(freqs), np.max(freqs))
    fft_freqs = np.arange(0, n_fft/2+1) / n_fft*f_s
    print(fft_freqs.shape)

    for idx in range(freqs.shape[0]-2):
        f_lo = freqs[idx]
        f_mid = freqs[idx+1]
        f_hi = freqs[idx+2]
        print(f_lo, f_mid, f_hi)

def melbank2(n_filters, n_fft, f_s, f_lo=0, f_hi=0.5, typ='log'):
    if typ == 'log':
        freq2scale = np.log10
        scale2freq = lambda x: np.power(10, x)
    elif typ == 'erb':
        freq2scale = frq2erb
        scale2freq = erb2frq
    elif typ == 'cat':
        freq2scale = frq2erb_cat
        scale2freq = erb2frq_cat

    # lo and hi filter frequencies in the desired units
    melfreq_lo, melfreq_hi = freq2scale([f_lo * f_s, f_hi * f_s])
    # fixed increment required to get from melfreq_lo to melfreq_hi in n_filters steps
    melinc = (melfreq_hi - melfreq_lo)/(n_filters-1)
    # add 2 extra increments... why? if using centres,you'd add half. surely?
    melfreq_lo, melfreq_hi = melfreq_lo-melinc, melfreq_hi+melinc
    # get fl_lo, fl_mid, fh_mid, fh_hi in Hz
    frq = scale2freq(melfreq_lo + np.array([0, 1, n_filters, n_filters+1]) * melinc)
    blim = frq * n_fft / f_s
    print('blim', blim)
    centre_freqs = melfreq_lo + np.arange(1, n_filters + 1) * melinc

    b1 = np.floor(blim[0]) + 1

    fn2 = np.floor(n_fft/2)
    b4 = np.min([fn2, np.ceil(blim[3])-1])
    print('b1', b1, 'fn2', fn2, 'b4', b4)
    print('centre_freqs', centre_freqs, centre_freqs.shape)

    sca = freq2scale(np.arange(b1, b4+1) * f_s / n_fft)
    pf = (sca - melfreq_lo) / melinc

    print('pf', pf.shape, np.min(pf), np.max(pf))

    if pf[0] < 0:
        pf = pf[1:]
        b1 = b1 + 1

    if pf[-1] >= n_filters + 1:
        pf = pf[:-1]
        b4 = b4 - 1

    fp = np.floor(pf)
    pm = pf - fp
    # the following are indices, so should be one less than matlab
    k4 = fp.shape[0] - 1
    try:
        k2 = np.where(fp > 0)[0][0]
    except IndexError:
        k2 = k4 + 1
    try:
        k3 = np.where(fp < n_filters)[0][-1]
        print('noerr')
    except IndexError:
        print('err')
        k3 = 0
    print('fp', fp[:10])
    print(np.min(pm), np.max(pm), pm.shape[0])
    print('k2',k2, 'k3',k3, 'k4',k4)

    r = np.concatenate((0+fp[:k3+1], fp[k2:k4+1])) # index?
    c = np.concatenate((np.arange(0, k3+1), np.arange(k2, k4+1))) # index
    v = np.concatenate((pm[:k3+1], 1-pm[k2:k4+1]))
    print('r', np.min(r), np.max(r), r.shape)
    print('c', np.min(c), np.max(c), c.shape)
    print('v', np.min(v), np.max(v), v.shape)
    mn = b1 + 1
    mx = b4 + 1
    print('mn', mn, 'mx', mx)
    if b1 < 0:
        c = np.abs(c+b1-1)-b1 + 1

    x = np.zeros((int(np.max(r)+1), int(np.max(c)+1)))
    for row, col, val in zip(r, c, v):
        # print(row,)
        x[int(row), int(col)] = val
    x = x / np.sum(x, axis=1)[:, np.newaxis]

    return x, centre_freqs, mn, mx

def melbank(n_filters, n_fft, f_s, f_lo=0, f_hi=0.5, typ='log'):
    '''
    p   number of filters in filterbank
    n   length of fft
    fs  sample rate in Hz
    fl  low end of the lowest filter as a fraction of fs [default = 0]
    fh  high end of highest filter as a fraction of fs [default = 0.5]
    w   any sensible combination of the following:
    '''
    sfact = 1

    mflh = np.array([f_lo, f_hi]) * f_s
    print(mflh)

    if typ == 'log':
        mflh = np.log10(mflh)
    elif typ == 'erb':
        mflh = frq2erb(mflh)
    elif typ == 'cat':
        mflh = frq2erb_cat(mflh)
    print(typ)
    print(mflh)
    melrng = mflh[1] - mflh[0]
    print(melrng)
    fn2 = np.floor(n_fft/2)

    melinc = melrng/(n_filters-1)
    print('melinc', melinc)
    mflh = [mflh[0]-melinc, mflh[1]+melinc]
    print('mflh', mflh)

    spc = mflh[0]+np.array([0, 1, n_filters, n_filters+1]) * melinc
    if typ == 'log':
        frq = np.power(10, spc)
    elif typ == 'erb':
        frq = erb2frq(spc)
    elif typ == 'cat':
        frq = erb2frq_cat(spc)
    print('spc', spc)
    print('frq', frq)

    blim = frq * n_fft / f_s
    print('blim', blim)

    mc = mflh[0] + np.arange(1, n_filters + 1) * melinc
    b1 = np.floor(blim[0])
    b4 = np.min([fn2, np.ceil(blim[3])])
    print('b1', b1, 'fn2', fn2, 'b4', b4)
    print(mc, mc.shape, b1, b4)

    frq = np.arange(b1, b4+1) * f_s / n_fft
    if typ == 'log':
        y = np.log10(frq)
    elif typ == 'erb':
        y = frq2erb(frq)
    elif typ == 'cat':
        y = frq2erb_cat(frq)

    pf = (y - mflh[0]) / melinc

    print(pf.shape, np.min(pf), np.max(pf))

    if pf[0] < 0:
        pf = pf[1:]
        b1 = b1 + 1

    if pf[-1] >= n_filters + 1:
        pf = pf[:-1]
        b4 = b4 - 1

    fp = np.floor(pf)
    print('fp', fp)
    pm = pf - fp
    print('pm', pm)
    # the following are indices, so should be one less than matlab
    k4 = fp.shape[0] - 1
    try:
        k2 = np.where(fp > 0)[0][0]
    except IndexError:
        k2 = k4 + 1
    try:
        k3 = np.where(fp < n_filters)[0][-1]
        print('noerr')
    except IndexError:
        print('err')
        k3 = 0
    print(fp[:10])
    print(np.min(pm), np.max(pm), pm.shape[0])
    print('k2',k2, 'k3', k3, 'k4',k4)

    r = np.concatenate((fp[:k3+1], fp[k2:k4+1])) # index?
    c = np.concatenate((np.arange(0, k3+1), np.arange(k2, k4+1))) # index
    v = np.concatenate((pm[:k3+1], 1-pm[k2:k4+1]))
    print('r', np.min(r), np.max(r), r.shape)
    print(r[:10])
    print('c', np.min(c), np.max(c), c.shape)
    print(c[:10])
    print('v', np.min(v), np.max(v), v.shape)
    print(v[:10])
    print(v[-10:])
    mn = b1 + 1
    mx = b4 + 1
    print('mn', mn, 'mx', mx)
    if b1 < 0:
        c = np.abs(c+b1-1)-b1 + 1

    x = np.zeros((int(np.max(r)+1), int(np.max(c)+1)))
    for row, col, val in zip(r, c, v):
        # print(row,)
        x[int(row), int(col)] = val
    x = x / np.sum(x, axis=1)[:, np.newaxis]

    return x, mc, mn, mx

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

def calc_CC_norm_old(R, yhat):
    '''
    Calculate CC_norm, CC_abs_CC_max of a y_td matrix where t is time and d are repeats
    '''
    N, T = R.shape[0], R.shape[1]
    y = np.mean(R, axis=0)
    Ey = np.mean(y)
    Eyhat = np.mean(yhat)
    Vy = np.sum(np.multiply((y-Ey), (y-Ey)))/T
    Vyhat = np.sum(np.multiply((yhat-Eyhat), (yhat-Eyhat)))/T
    Cyyhat = np.sum(np.multiply((y-Ey), (yhat-Eyhat)))/T
    SP = (np.var(np.sum(R, axis=0), ddof=1)-np.sum(np.var(R, axis=1, ddof=1)))/(N*(N-1))
    print(SP)
    CCabs = Cyyhat/np.sqrt(Vy*Vyhat)
    CCnorm = Cyyhat/np.sqrt(SP*Vyhat)
    CCmax = np.sqrt(SP/Vy)
    if SP <= 0:
        print('SP less than or equal to zero - CCmax and CCnorm cannot be calculated.')
        CCnorm = np.nan
        CCmax = 0
    return CCnorm, CCabs, CCmax
