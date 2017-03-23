# Filtering functions.
import numpy as np

def window_signal(i,window,T,signal,extract=True):
    """
    Slow way of getting view of each shifted window. Each row is the window shifted over.
    2017-03-23
    
    Params:
    -------
    window (ndarray)
    shift (int)
    T (int)
        Total duration of signal.
    extract (bool=True)
        If true, only return the nonzero part of window multiplied with the signal.
    """
    if extract:
        return window*signal[i-lw//2:i+lw//2+1]
    else:
        swindow = np.zeros((T))
        swindow[i-lw//2:i+lw//2+1] = window*signal[i-lw//2:i+lw//2+1]
    return swindow

def _shifted_window_view(window,shift,T):
    """
    Slow way of getting view of each shifted window. Each row is the window shifted over.
    2017-03-23
    
    Params:
    -------
    window (ndarray)
    shift (int)
    T (int)
        Total duration of signal.
    """
    views = []
    lw = len(window)
    for i in xrange(lw//2,T-lw//2,dx):
        swindow = np.zeros((T))
        swindow[i-lw//2:i+lw//2+1] = window
        views.append( swindow )
    return np.vstack(views)

def shifted_window_weights(window,shift,T):
    """
    Slow way of getting the weight at each entry in signal as window is being shifted over. Assumes that the 
    window is first placed as far as possible on the left hand side (without going over boundary) and shifted
    over til no more FULL windows can be placed into the array of length T. This may not be what you
    want in general!
    2017-03-23
    
    Params:
    -------
    window (ndarray)
    shift (int)
    T (int)
        Total duration of signal.
    """
    wnorm = np.zeros((T))
    lw = len(window)
    for i in xrange(lw//2,T-lw//2,dx):
        wnorm[i-lw//2:i+lw//2+1] += window
    return wnorm

def moving_freq_filt(s,window=61,windowType=('gaussian',20),cutoffFreq=5,sampleFreq=60,
                     mx_filter_rows=1000):
    """
    Moving frequency filter using Butterworth lowpass filter. First, construct windowed input as given
    window type is dragged across. Frequency filter each of those samples and then add them all up 
    back together to get the filtered signal.
    2017-03-19
    
    Params:
    -------
    s (ndarray)
        1d signal
    window (int=61)
        Window width.
    windowType (tuple)
        ('Gaussian',20) as input to scipy.signal.get_window()
    cutoffFreq (float=5)
        Cutoff frequency for butterworth filter.
    sampleFreq (float=60)
        Sampling frequency of input signal.
    mx_filter_rows (int=1000)
        Maximum number of rows to filter at once. This is limited by memory.
    """
    assert (window%2)==1, "Window width must be odd."
    from scipy.signal import get_window
    
    T=len(s)
    swindow=np.zeros((T))  # windowed input

    # Normalize the window. Each point gets hit with every part of window once (except boundaries).
    window = get_window(windowType,window)
    window /= window.sum()

    # Extract subsets of data while window is moving across.
    def f(i):
        return butter_lowpass_filter( _moving_window(i,s,window),
                                      cutoffFreq,sampleFreq )

    # Given memory constraints, don't filter everything at once.
    pool = mp.Pool(mp.cpu_count())
    swindow = np.zeros((T))
    for i in xrange(0,T-mx_filter_rows,mx_filter_rows):
        swindow += np.vstack( pool.map(f,range(i,i+mx_filter_rows)) ).sum(0)
    if (i+mx_filter_rows)<T:
        if (i+mx_filter_rows)==(T-1):
            swindow += pool.map(f,range(i+mx_filter_rows,T))
        else:
            swindow += np.vstack( pool.map(f,range(i+mx_filter_rows,T)) ).sum(0)
    pool.close()

    #swindow = butter_lowpass_filter( swindow,
    #                                 cutoffFreq,
    #                                 sampleFreq,
    #                                 axis=1 ).sum(0)
    return swindow

