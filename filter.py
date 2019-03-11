# Filtering functions.

from .data_access import *
import numpy as np
import multiprocess as mp
from numpy import fft

def spectrogram(s,window,shift,fs=1,npadding=0,padval=0.):
    """
    This does not account for the proper normalization of the windowing function such that the original signal
    can be reconstituted. This would have to be done with shifted_window_weights().
    2017-03-24

    Params:
    -------
    s (ndarray)
    window (ndarray)
    shift (int)
        Number of indices to shift window as taking moving spectrogram.
    fs (float=1)
        Sampling rate.
    npadding (int=0)
    padval (float=0.)

    Returns:
    --------
    f (ndarray)
        Frequencies.
    t (ndarray)
    spec
        (n_freq,n_time)
    """
    lw = len(window)
    f = fft.fftfreq(len(window),d=1/fs)
    
    # Define time, accounting for padding.
    assert npadding<=(lw//2)
    s = np.concatenate(( np.zeros((npadding))+padval,s,np.zeros((npadding))+padval ))
    t = (1/fs)*np.arange( lw//2-npadding,len(s)-(lw//2-npadding),shift )
    
    spec = np.zeros((len(f),len(t)),dtype=np.complex64)
    #windowWeights = shifted_window_weights(window,shift,len(s),offset=lw//2)
    
    # Remember that the offset should be kept at lw//2 when padded because the only thing that the padding
    # does is to shift the beginning of the windowing over but you still want a full window to fit at the left
    # and right boundaries.
    for counter,i in enumerate(range(lw//2,len(s)-lw//2,shift)):
        spec[:,counter] = fft.fft( window_signal(i,window,s) )

    return f,t,spec

def window_signal(i,window,signal,extract=True,return_index=False):
    """
    Slow way of getting view one shifted window.
    2017-03-23
    
    Params:
    -------
    window (ndarray)
    shift (int)
    extract (bool=True)
        If true, only return the nonzero part of window multiplied with the signal.
    return_index (bool=False)
        If true, return the index range that we are windowing. This only applies if extract is True.

    Returns:
    --------
    swindow (ndarray)
        Windowed view of signal.
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
    
    for i in range(offset[0],lw//2,shift):
        swindow = np.zeros((T))
        swindow[:i+lw//2+1] = window[-(i-lw//2):]
        views.append( swindow )

    for i in range(lw//2+(lw//2)%shift,T-lw//2,shift):
        swindow = np.zeros((T))
        swindow[i-lw//2:i+lw//2+1] = window
        views.append( swindow )

    for i in range(T-lw//2+(lw//2)%shift,T-offset[1]):
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
    # Off by one index error on smaller arrays...358. wnorm is too small by one. or rather window is too large by one.
    for i in range(offset[0],lw//2,shift):
        wnorm[:i+lw//2+1] += window[-(i-lw//2):]

    for i in range(lw//2+(lw//2)%shift,T-lw//2,shift):
        wnorm[i-lw//2:i+lw//2+1] += window

    for i in range(T-lw//2+(lw//2)%shift,T-offset[1]):
        wnorm[i-lw//2:] += window[:-(lw//2-(T-i))-1]

    return wnorm

def moving_freq_filt(s,axis=-1,**kwargs):
    """
    Moving frequency filter using Butterworth lowpass filter. First, construct windowed input as given
    window type is dragged across. Frequency filter each of those samples and then add them all up 
    back together to get the filtered signal.

    Wrapper for _moving_freq_fil()
    2017-03-26
    
    Params:
    -------
    s (ndarray)
        1d or 2d signal
    axis (int=-1)
        Axis along which data extends in time. For example, if each time sample is a new row, then axis should
        be 0.
    window (int=61)
        Window width.
    window_type (tuple)
        ('Gaussian',20) as input to scipy.signal.get_window()
    window_shift (int=1)
        Not yet implemented.
    filter_type (str=None)
        Default is 'butter' which is a Butterworth lowpass filter.
    cutoff_freq (float=5)
        Cutoff frequency for butterworth filter.
    pass_freq (float=None)
        Pass frequency for Gaussian bandpass filter.
    bandwidth (float=None)
        Standard deviation for width of Gaussian bandpass filter in units of index.
    sample_freq (float=60)
        Sampling frequency of input signal.
    mx_filter_rows (int=100)
        Maximum number of rows to filter at once. This is limited by memory.

    Returns:
    --------
    filtereds (ndarray)
        Filtered signal. Same size as given signal.
    """
    # Multi-dimensional data.
    if s.ndim>1:
        if axis==0:
            s = s.T
            return np.vstack([ _moving_freq_filt(x,**kwargs) for x in s]).T
        elif axis==-1 or axis==1:
            return np.vstack([ _moving_freq_filt(x,**kwargs) for x in s])

    # Single dimensional data.
    return _moving_freq_filt(s,**kwargs)
        

def _moving_freq_filt(s,
                      window=201,
                      window_type=('gaussian',20),
                      window_shift=1,
                      filter_type=None,
                      cutoff_freq=5,
                      pass_freq=None,
                      bandwidth=None,
                      sample_freq=60,
                      mx_filter_rows=100):
    """
    Called in moving_freq_filt().

    Moving frequency filter using Butterworth lowpass filter. First, construct windowed input as given
    window type is dragged across. Frequency filter each of those samples and then add them all up 
    back together to get the filtered signal.
    
    Params:
    -------
    s (ndarray)
        1d signal
    window (int=61)
        Window width.
    window_type (tuple)
        ('Gaussian',20) as input to scipy.signal.get_window()
    window_shift (int=1)
        Not yet implemented.
    filter_type (str=None)
        Default is 'butter' which is a Butterworth lowpass filter.
    cutoff_freq (float=5)
        Cutoff frequency for butterworth filter.
    pass_freq (float=None)
        Pass frequency for Gaussian bandpass filter.
    bandwidth (float=None)
        Standard deviation for width of Gaussian bandpass filter in units of index.
    sample_freq (float=60)
        Sampling frequency of input signal.
    mx_filter_rows (int=100)
        Maximum number of rows to filter at once. This is limited by memory.
    """
    window_shift = 1  # Have not yet implemented different window shifts
    assert (window%2)==1, "Window width must be odd."
    from scipy.signal import get_window
    filter_type = filter_type or 'butter'
    
    T = len(s)
    swindow = np.zeros((T))  # windowed input
    filtfun = {'butter':(lambda x: butter_lowpass_filter(x,cutoff_freq,sample_freq)),
               'single':(lambda x: single_freq_pass_filter(x,pass_freq,bandwidth,sample_freq))
              }[filter_type]
    
    # Normalize the window. Each point gets hit with every part of window once (except boundaries).
    window = get_window(window_type,window)
    lw = len(window)
    windowWeights = shifted_window_weights(window,window_shift,T,offset=[0,0])

    # Extract subsets of data while window is moving across.
    def f(i):
        view,ix = window_signal(i,window,s,extract=True,return_index=True)
        filteredView = filtfun(view)
        filtereds = np.zeros((T))
        filtereds[ix[0]:ix[1]] = filteredView
        return filtereds
    
    # Given memory constraints, don't filter everything at once.
    pool = mp.Pool(mp.cpu_count())
    swindow = np.zeros((T))
    for i in range(0,T-mx_filter_rows,mx_filter_rows):
        swindow += np.vstack( pool.map(f,list(range(i,i+mx_filter_rows))) ).sum(0)
    if (i+mx_filter_rows)<(T-1):
        if (i+mx_filter_rows)==(T-1):
            swindow += pool.map(f,list(range(i+mx_filter_rows,T)))
        else:
            swindow += np.vstack( pool.map(f,list(range(i+mx_filter_rows,T))) ).sum(0)
    pool.close()

    return swindow/windowWeights

def single_freq_pass_filter(x,pass_freq,bandwidth,sample_freq):
    """
    Single frequency filter using a Gaussian frequency filter (signal assumed to be real).
    
    Params:
    -------
    x (ndarray)
    pass_freq (float)
    sample_freq (float)

    Returns:
    --------
    filteredx (ndarray)
        Bandpass filtered signal. Only the real part is taken because signal is assumed to be real.
    """
    from scipy.stats import norm

    freq = fft.fftfreq(len(x),d=1/sample_freq)
    spec = fft.fft(x)*norm.pdf(freq,pass_freq,bandwidth)*np.sqrt(2*np.pi)*bandwidth
    return fft.ifft(spec).real

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

def smooth(x,filtertype='moving_butter',filterparams='default',
           moving_freq_kwargs={}):
    """
    Smooth multidimensional curve. Currently, using Savitzy-Golay on each
    dimension independently. Ideally, I would implememnt some sort of smoothing
    algorithm that accounts for relationships between the different dimensions.
    2017-01-24

    Params:
    -------
    x (ndarray)
        n_samples x n_dim
    filtertype (str='sav')
        'sav', 'butter', 'moving_butter'
    filterparams (dict)
        Savitzy-Golay: (window, order) typically {'window':61,'order':4}
        Butterworth: (cutoff, fs) typically {'cutoff':10,'fs':60}
        Moving Butterworth: (cutoff, fs) typically {'cutoff':10,'fs':60}
    moving_freq_kwargs (dict)
    """
    if filtertype=='sav':
        from scipy.signal import savgol_filter
        if type(filterparams) is str:
            if filterparams=='default':
                filterparams={'window':61,'order':4}
            else:
                raise NotImplementedError
        elif not type(filterparams) is dict:
            raise NotImplementedError

        if x.ndim==1:
            return savgol_filter(x,
                                 filterparams['window'],filterparams['order'])
        
        xfiltered=np.zeros_like(x)
        for i in range(x.shape[1]):
            xfiltered[:,i]=savgol_filter(x[:,i],window,order)
    elif filtertype=='butter':
        if type(filterparams) is str:
            if filterparams=='default':
                filterparams={'cutoff':10,'fs':60}
            elif filterparams=='120':
                filterparams={'cutoff':10,'fs':120}
            else:
                raise NotImplementedError
        elif not type(filterparams) is dict:
            raise NotImplementedError

        if x.ndim==1:
            axis=-1
        else:
            axis=0
        xfiltered=butter_lowpass_filter(x,
                                        filterparams['cutoff'],
                                        filterparams['fs'],
                                        axis=axis) 
    elif filtertype=='moving_butter':
        if type(filterparams) is str:
            if filterparams=='default':
                filterparams={'cutoff':10,'fs':60}
            elif filterparams=='120':
                filterparams={'cutoff':10,'fs':120}
            else:
                raise NotImplementedError
        elif not type(filterparams) is dict:
            raise NotImplementedError

        if x.ndim==1:
            xfiltered=moving_freq_filt(x,
                                       cutoff_freq=filterparams['cutoff'],
                                       sample_freq=filterparams['fs'],
                                       **moving_freq_kwargs)
        else:
            xfiltered = moving_freq_filt(x,
                                        cutoff_freq=filterparams['cutoff'],
                                        sample_freq=filterparams['fs'],
                                        axis=0,
                                        **moving_freq_kwargs)
    else: raise Exception("Invalid filter option.")
    return xfiltered

def detrend(x,inplace=False,return_fit=False):
    """
    Detrend by fitting a low order polynomial to the data and subtracting it.

    Params:
    -------
    x (ndarray)
        1d.
    inplace (bool=False)
    return_fit (bool=False)
    """
    T = np.arange(len(x))
    p = np.polyfit(T,x,3)
    if inplace:
        x[:] -= np.polyval(p,T)
        if return_fit:
            return p
    else:
        if return_fit:
            return x - np.polyval(p,T),p
        return x - np.polyval(p,T)

