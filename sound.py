'''
Basic sound functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram as ss_spec

def tdt50k():
    '''
    TDT "50k" sample rate
    '''
    return 48828.125

def amp2level(amp):
    '''
    Convert an amplitude multiplier to level in dB.
    E.g. a multiplier of 10 is +20dB
    '''
    return 20*np.log10(amp)

def level2amp(dB):
    '''
    Convert a dB difference to amplitude.
    E.g. a difference of +10dB is a multiplier of 3.16
    '''
    return 10**(dB/20)

def rms2dBSPL(rms_Pa):
    '''
    Convert a sound RMS in Pascals to dB SPL.
    E.g. 1 Pa RMS = 94dB
    '''
    return amp2level(rms_Pa)+94

def dBSPL2rms(dBSPL):
    '''
    Convert dBSPL to RMS in Pascals
    E.g. 94dB = 1 Pa RMS
    '''
    return level2amp(dBSPL-94)

def puretone(fs, n_samples, freq, level_dB=94, phase=0):
    '''
    Generate a pure tone
    '''
    t = np.arange(n_samples) * 1/fs
    return np.sin(2*np.pi*freq*t + phase) * np.sqrt(2) * dBSPL2rms(level_dB)

def freq_sweep(fs, n_samples, f_min, f_max, level_dB=94, phase=0):
    '''
    Generate a frequency sweep
    '''
    t = np.arange(n_samples) * 1/fs
    freq = np.linspace(f_min, f_max, n_samples)
    return np.sin(2*np.pi*freq*t + phase) * np.sqrt(2) * dBSPL2rms(level_dB)

def cosramp_on(n_samples, ramp_samples=None):
    '''
    Ramp on - total length n_samples, ramp length ramp_samples
    '''
    if ramp_samples is None:
        ramp_samples = n_samples
    t = np.minimum(np.arange(n_samples), ramp_samples)
    return np.sin(np.pi/2/ramp_samples*t)

def cosramp_off(n_samples, ramp_samples=None):
    '''
    Ramp off - total length n_samples, ramp length ramp_samples
    '''
    if ramp_samples is None:
        ramp_samples = n_samples
    return cosramp_on(n_samples, ramp_samples)[::-1]

def cosramp_onoff(n_samples, ramp_samples):
    '''
    Ramp on and off - total length n_samples, ramp lengths ramp_samples
    '''
    r = cosramp_on(n_samples, ramp_samples)
    return r * r[::-1]

def spectrogram(*args, **kwargs):
    '''
    This is just to help find the scipy spectrogram function
    '''
    return ss_spec(*args, **kwargs)

def show_spectrogram(*args, **kwargs):
    '''
    Show spectrogram using pyplot
    '''
    f, t, s = ss_spec(*args, **kwargs)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(s, origin='lower', extent=[np.min(t), np.max(t), np.min(f), np.max(f)])
    ax.set_aspect((np.max(t)-np.min(t))/(np.max(f)-np.min(f))/2)