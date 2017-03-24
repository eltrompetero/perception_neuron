# Filtering functions.
import numpy as np
import multiprocess as mp

def window_signal(i,window,signal,extract=True,return_index=False):
    """
    Slow way of getting view of each shifted window. Each row is the window shifted over.
    2017-03-23
    
    Params:
    -------
    window (ndarray)
    shift (int)
    extract (bool=True)
        If true, only return the nonzero part of window multiplied with the signal.
    return_index (bool=False)
        If true, return the index range that we are windowing. This only applies if extract is True.
    """
    lw,T = len(window),len(signal)
    if extract:
        if i<(lw//2):
            swindow = window[-(i-lw//2):]*signal[:i+lw//2+1]
            if return_index:
                return swindow,(0,i+lw//2+1)
        elif (lw//2)<=i<(T-lw//2):
            swindow = window*signal[i-lw//2:i+lw//2+1]
            if return_index:
                return swindow,(i-lw//2,i+lw//2+1)
        else:
            swindow = window[:-(lw//2-(T-i))-1]*signal[i-lw//2:]
            if return_index:
                return swindow,(i-lw//2,T)
        return swindow
    else:
        swindow = np.zeros_like(signal)
        
        if i<(lw//2):
            swindow[:i+lw//2+1] = window[-(i-lw//2):]*signal[:i+lw//2+1]
        elif (lw//2)<=i<(T-lw//2):
            swindow[i-lw//2:i+lw//2+1] = window*signal[i-lw//2:i+lw//2+1]
        else:
            swindow[i-lw//2:] = window[:-(lw//2-(T-i))-1]*signal[i-lw//2:]
        return swindow

def _shifted_window_view(window,shift,T,offset=None):
    """
    Slow way of getting view of each shifted window. Each row is the window shifted over. You can use this to
    check that window_signal() is implemented correctly. Some example test code:

    T = 20
    window = get_window(('gaussian',1),5)
    dx = 4
    plt.matshow(shifted_windows_view(window,dx,T))

    plt.plot(shifted_windows_view(window,dx,T).sum(0))
    plt.plot(shifted_windows_weights(window,dx,T))

    2017-03-23
    
    Params:
    -------
    window (ndarray)
    shift (int)
    T (int)
        Total duration of signal.
    offset (tuple or int)
        Offset on left and right side.
    """
    views = []
    lw = len(window)
    if offset is None:
        offset = [lw//2]*2
    elif type(offset) is int:
        offset = [offset]*2
    
    for i in xrange(offset[0],lw//2,shift):
        swindow = np.zeros((T))
        swindow[:i+lw//2+1] = window[-(i-lw//2):]
        views.append( swindow )

    for i in xrange(lw//2+(lw//2)%shift,T-lw//2,shift):
        swindow = np.zeros((T))
        swindow[i-lw//2:i+lw//2+1] = window
        views.append( swindow )

    for i in xrange(T-lw//2+(lw//2)%shift,T-offset[1]):
        swindow = np.zeros((T))
        swindow[i-lw//2:] = window[:-(lw//2-(T-i))-1]
        views.append( swindow )

    return np.vstack(views)

def shifted_window_weights(window,shift,T,offset=None):
    """
    Slow way of getting the weight at each entry in signal as window is being shifted over. Default is that
    the window is first placed as far as possible on the left hand side (without going over boundary) and
    shifted over til no more FULL windows can be placed into the array of length T. This may not be what you
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
    if offset is None:
        offset = [lw//2]*2
    elif type(offset) is int:
        offset = [offset]*2

    for i in xrange(offset[0],lw//2,shift):
        wnorm[:i+lw//2+1] += window[-(i-lw//2):]

    for i in xrange(lw//2+(lw//2)%shift,T-lw//2,shift):
        wnorm[i-lw//2:i+lw//2+1] += window

    for i in xrange(T-lw//2+(lw//2)%shift,T-offset[1]):
        wnorm[i-lw//2:] += window[:-(lw//2-(T-i))-1]

    return wnorm

def moving_freq_filt(s,window=201,windowType=('gaussian',20),windowShift=1,cutoffFreq=5,sampleFreq=60,
                     mx_filter_rows=1000):
    """
    Moving frequency filter using Butterworth lowpass filter. First, construct windowed input as given
    window type is dragged across. Frequency filter each of those samples and then add them all up 
    back together to get the filtered signal.
    2017-03-23
    
    Params:
    -------
    s (ndarray)
        1d signal
    window (int=61)
        Window width.
    windowType (tuple)
        ('Gaussian',20) as input to scipy.signal.get_window()
    windowShift (int=1)
        Not yet implemented.
    cutoffFreq (float=5)
        Cutoff frequency for butterworth filter.
    sampleFreq (float=60)
        Sampling frequency of input signal.
    mx_filter_rows (int=1000)
        Maximum number of rows to filter at once. This is limited by memory.
    """
    windowShift = 1  # Have not yet implemented different window shifts
    assert (window%2)==1, "Window width must be odd."
    from scipy.signal import get_window
    
    T = len(s)
    swindow = np.zeros((T))  # windowed input

    # Normalize the window. Each point gets hit with every part of window once (except boundaries).
    window = get_window(windowType,window)
    lw = len(window)
    windowWeights = shifted_window_weights(window,windowShift,T,offset=[0,0])

    # Extract subsets of data while window is moving across.
    def f(i):
        view,ix = window_signal(i,window,s,extract=True,return_index=True)
        filteredView= butter_lowpass_filter( view, cutoffFreq,sampleFreq )
        filtereds = np.zeros((T))
        filtereds[ix[0]:ix[1]] = filteredView
        return filtereds
    
    # Given memory constraints, don't filter everything at once.
    pool = mp.Pool(mp.cpu_count())
    swindow = np.zeros((T))
    for i in xrange(0,T-mx_filter_rows,mx_filter_rows):
        swindow += np.vstack( pool.map(f,range(i,i+mx_filter_rows)) ).sum(0)
    if (i+mx_filter_rows)<(T-1):
        if (i+mx_filter_rows)==(T-1):
            swindow += pool.map(f,range(i+mx_filter_rows,T))
        else:
            swindow += np.vstack( pool.map(f,range(i+mx_filter_rows,T)) ).sum(0)
    pool.close()

    #swindow = butter_lowpass_filter( swindow,
    #                                 cutoffFreq,
    #                                 sampleFreq,
    #                                 axis=1 ).sum(0)
    return swindow/windowWeights

def butter_lowpass(cutoff,fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    cutoff /= nyq
    b, a = butter( order, cutoff )
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5, axis=0, **kwargs):
    """
    Forward-backward call to Butterworth filter. This applies the filter in the forwards direction and then
    backwards and the result is in phase with the data.
    2015-10-18

    Params:
    -------
    data (ndarray)
    cutoff (float)
        max frequency in Hz corresponding to frequency at which gain goes to -3dB, or 1/sqrt(2)
    fs (float)
        sample frequency in Hz
    order (5,int)
        order of Buttworth filter
    """
    from scipy.signal import filtfilt
    b,a = butter_lowpass(cutoff,fs,order)
    return filtfilt( b,a,data,axis=axis, **kwargs)

def butter_plot(ax, b, a, fs):
    """Show plot of filter gain.
    2015-09-09
    """
    w, h = signal.freqz(b, a)
    ax.semilogx(w*.5*fs/np.pi, 20 * np.log10(abs(h)))
    ax.set(title='Butterworth filter frequency response',
           xlabel='Frequency [Hz]',
           ylabel='Amplitude [dB]',ylim=[-5,1])
    #ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')
    ax.axvline(10, color='green') # cutoff frequency

