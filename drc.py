'''
Dynamic random chords
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np

def levels2drc(f_s, freqs, levels, chord_dur, ramp_dur):
    '''
    Create a DRC waveform from a grid of levels.
    Inputs:
    f_s - sample rate
    freqs - the frequencies of the tones to use
    grid of levels - n_freqs x n_chords
    chord_duration - in seconds
    ramp_duration - in seconds

    Output:
    drc - structure containing waveform as drc.snd and other information

    Differences between this and the last version (very similar to Neil's classic DRCs)
    * at every transition (inc start/end), there is a cosine ramp in amplitude, not level
    * the first chord is the same as others
    * the end is slightly different

    NB THIS VERSION OUTPUTS SOUNDS IN PASCALS, SO THAT 1 UNIT RMS = 94DB
    '''
    print('UNTESTED')

    drc = {'f_s': f_s,
           'freqs': freqs,
           'levels': levels,
           'chord_dur': chord_dur,
           'ramp_dur': ramp_dur,
           'n_chords': levels.shape[1],
           'n_freqs': levels.shape[0]
          }

    # samples on which chords will start
    drc['chord_start_times'] = np.arange(0, drc['n_chords']+1)*drc['chord_dur']
    drc['chord_start_samples'] = np.round(drc['chord_start_times']*drc['f_s'])

    # make a standard envelope for a single chord (ramp up then hold)
    drc['ramp_samples'] = np.int(np.round(drc['ramp_dur'] * drc['f_s']))
    rt = np.linspace(-np.pi/2, np.pi/2, drc['ramp_samples'])
    cosramp = np.sin(rt)/2+0.5
    max_chordlen = np.int(np.max(np.diff(drc['chord_start_samples'])))
    chord_env = np.concatenate((cosramp, np.ones(max_chordlen-drc['ramp_samples'])))

    # time vector for whole stimulus
    drc['total_samples'] = np.max(drc['chord_start_samples'])-1 + drc['ramp_samples']
    drc['t'] = np.arange(0, drc['total_samples']+1)/drc['f_s']

    # convert level to amplitude
    drc['amplitudes'] = 10**((drc['levels']-94)/20)

    # build up stimulus, one tone at a time
    snd = None

    for freq_idx in range(0, drc['n_freqs']):
        print('.', end='')

        # make carrier sinusoid
        freq = drc['freqs'][freq_idx]
        carrier = np.sin(2*np.pi*freq*drc['t'])*np.sqrt(2) # RMS = 1

        # make envelope
        env = []

        last_level = 0

        # ramp from the last amplitude to the current one,
        # then hold at the current amplitude

        for chord_idx in range(0, drc['n_chords']):
            level = drc['amplitudes'][freq_idx, chord_idx]
            ln = drc['chord_start_samples'][chord_idx+1] - \
                drc['chord_start_samples'][chord_idx]

            env.append(last_level+chord_env[:int(ln)]*(level-last_level))
            last_level = level

        # final ramp down to zero
        level = 0
        env.append(last_level+cosramp*(level-last_level))

        env = np.concatenate(env)

        # superpose the frequency channels on one another to
        # get a single sound vector
        if snd is None:
            snd = carrier * env
        else:
            snd = snd + carrier * env

    drc['example_envelope'] = env
    drc['snd'] = snd

    return drc

def make_example_drc():
    stim = {}
    stim['f_s'] = 24414*2 # floor(TDT50K)
    stim['freq_min'] = 500
    stim['freq_step'] = 2**(1/6)
    stim['freq_n'] = 34

    stim['chord_dur'] = 25/1000
    stim['ramp_duration'] = 5/1000

    # get frequencies of tones
    stim['freq_multipliers'] = np.arange(stim['freq_n'])
    stim['freqs'] = stim['freq_min']*stim['freq_step']**stim['freq_multipliers']

    # l = load('./test/grid.40Hz.gain_time_course.chord_fs.40.token.1.mat');
    grid = np.random.random((stim['freq_n'], 100))
    drc = levels2drc(stim['f_s'], stim['freqs'], grid, stim['chord_dur'], stim['ramp_duration'])

    stim['drc'] = drc
    stim['grid'] = grid

    return stim
