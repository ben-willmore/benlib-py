'''
Cochleagram
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from scipy.signal import spectrogram

def frq2erb(frq):
    '''
    See voicebox
    FRQ2ERB  Convert Hertz to ERB frequency scale ERB=(FRQ)
    '''
    frq = np.array(frq)
    g = np.abs(frq)
    erb = 11.17268 * np.sign(frq) *np.log(1+46.06538*g/(g+14678.49))
    bnd = 6.23e-6 * g**2 + 93.39e-3 * g + 28.52
    return erb, bnd

def erb2frq(erb):
    '''
    See voicebox
    ERB2FRQ  Convert ERB frequency scale to Hertz FRQ=(ERB)
    '''
    erb = np.array(erb)
    frq = np.sign(erb) * (676170.4 * (47.06538-np.exp(0.08950404*np.abs(erb)))**(-1) - 14678.49)
    bnd = 6.23e-6 * frq**2 + 93.39e-3 * np.abs(frq) + 28.52
    return erb, bnd

def frq2erb_cat(frq):
    '''
    Equivalent of frq2erb for cat
    '''
    frq = np.array(frq)
    erb = 1000 * 13.7 * (frq**0.362)
    return erb

def erb2frq_cat(erb):
    '''
    Equivalent of erb2frq for cat
    '''
    frq = erb / ((13.7 * 1000))**(1./0.362)
    return frq

def melbank(n_filters, n_fft, f_s, f_lo, f_hi, spacing='log'):
    '''
    Melbank. Frequencies are equally spaced in the specified space (log etc)
    The middle of one filter is the bottom of the next and the top of the previous
    one, and they are triangular in shape.
    '''
    if spacing == 'log':
        freq2scale = np.log10
        scale2freq = lambda x: np.power(10, x)
    elif spacing == 'erb':
        freq2scale = frq2erb
        scale2freq = erb2frq
    elif spacing == 'cat':
        freq2scale = frq2erb_cat
        scale2freq = erb2frq_cat

    # lo and hi filter frequencies in the desired units
    melfreq_lo, melfreq_hi = freq2scale([f_lo, f_hi])

    # fixed increment required to get from melfreq_lo to melfreq_hi in n_filters steps
    melinc = (melfreq_hi - melfreq_lo)/(n_filters-1)

    # add 2 extra increments for the top and bottom because we specified the
    # centre frequencies
    melfreq_lo, melfreq_hi = melfreq_lo-melinc, melfreq_hi+melinc

    # filter frequencies. freqs[0:-2] are the bottoms of the filters,
    # freqs[1:-1] are the middles, freqs[2:] are the tops
    freqs = scale2freq(melfreq_lo + np.arange(0, n_filters+2) * melinc)

    # artificially long fft_freqs to avoid clipping
    fft_freqs = np.arange(0, n_fft+1)/n_fft*f_s

    filters = []
    for idx in range(freqs.shape[0]-2):
        filt = np.zeros(fft_freqs.shape)
        [f_lo, f_mid, f_hi] = freqs[idx:idx+3]
        f_lo_idx = np.where(fft_freqs > f_lo)[0][0]
        f_mid_idx = np.where(fft_freqs > f_mid)[0][0]
        f_hi_idx = np.where(fft_freqs > f_hi)[0][0]

        # ramp up, then down again, about centre frequency
        if f_mid_idx > f_lo_idx:
            filt[f_lo_idx:f_mid_idx+1] = \
                np.arange(f_mid_idx+1-f_lo_idx)/(f_mid_idx-f_lo_idx)
        if f_hi_idx > f_mid_idx:
            filt[f_mid_idx:f_hi_idx+1] = \
                1 - np.arange(f_hi_idx+1-f_mid_idx)/(f_hi_idx-f_mid_idx)

        # clip filter to appropriate length
        filt = filt[:int(n_fft/2+1)]
        if np.sum(filt) > 0:
            filt = filt / np.sum(filt)
        filters.append(filt)
    return np.stack(filters, axis=0), freqs[1:-1]

def cochleagram(y_t, fs_hz, dt_ms, spacing='log', freq_info=None):
    '''
    Melbank-based log cochleagram. Need to check it thoroughly against
    matlab, and sort out threshold which is arbitrary
    '''
    params = {'spacing': spacing,
              'fs_hz': fs_hz,
              'threshold': -40}
    if freq_info is not None:
        params['f_min'], params['f_max'], params['n_f'] = freq_info

    if spacing == 'log':
        params['nfft_mult'] = 4
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 31

    elif spacing == 'cat-erb':
        params['nfft_mult'] = 1
        if freq_info is None:
            params['f_min'] = 1000
            params['f_max'] = 32000
            params['n_f'] = 23

    # get actual dt (which is an integer number of samples)
    dt_sec_nominal = dt_ms/1000
    dt_bins = np.round(dt_sec_nominal*params['fs_hz'])
    params['dt_sec'] = dt_bins/params['fs_hz']

    # get window, overlap sizes
    t_window_bins = dt_bins * 2
    params['t_window_sec'] = t_window_bins/params['fs_hz']
    t_overlap_bins = t_window_bins - dt_bins

    # melbank
    while True:
        [filts, melfreqs] = melbank(
            params['n_f'], int(t_window_bins*params['nfft_mult']),
            fs_hz, f_lo=params['f_min'], f_hi=params['f_max'],
            spacing=spacing)

        if np.all(np.sum(filts[:10, :], axis=1) > 0):
            break
        params['nfft_mult'] = params['nfft_mult'] * 2
        print('boosting nfft_mult to %d' % params['nfft_mult'])

    [freqs, t, spec] = spectrogram(y_t, fs=fs_hz, nperseg=int(t_window_bins),
                                   noverlap=int(t_overlap_bins),
                                   nfft=int(t_window_bins*params['nfft_mult']))

    X_ft = 10*np.maximum(np.log10(np.dot(filts, spec).shape), params['threshold'])
    params['melbank'] = filts
    params['melfreqs'] = melfreqs
    params['spectrogram'] = spec
    params['spectrogram_freqs'] = freqs
    params['spectrogram_t'] = t
    return X_ft, params
