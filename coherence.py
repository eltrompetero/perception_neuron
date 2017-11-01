# ================================================================================================ # 
# Functions for calculating coherence measure.
# Author: Eddie Lee, edl56@cornell.edu
# ================================================================================================ # 

from __future__ import division
import numpy as np
from scipy.signal import coherence
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF
from fastdtw import fastdtw


def precompute_coherence_nulls(v,t0,windowDuration,pool,
        sampling_rate=30,n_iters=100):
    """
    Calculate coherence values for trajectory with many samples of white noise. 
    
    This uses multiprocess to speed up calculation.
    
    Parameters
    ----------
    v : function
        Given times, return signal.
    t0 : ndarray
        Times at which windows begin for calculating nulls.
    windowDuration : float
        Window duration in seconds
    pool : multiprocess.Pool
        Pool for parallelizing computation.
        
    Returns
    -------
    One tuple for each x,y,z axis with
    f : ndarray
        Frequencies at which coherence was calculated
    coh_mean : ndarray
        Mean of coherence over random noise samples.
    coh_std : ndarray
        Std of coherence over random noise samples.
    """
    import multiprocess as mp
    
    def f(t0):
        # Data to analyize.
        t = t0+np.arange(windowDuration*sampling_rate)/sampling_rate
        v_ = v(t)
        
        # Coherence null values for each axis independently.
        Cnullx,Cnully,Cnullz = [],[],[]
        for i in xrange(n_iters):
            fx,cwtcohx = cwt_coherence_auto_nskip(v_[:,0],np.random.normal(size=len(v_)),
                                         sampling_period=1/sampling_rate,period_multiple=3)
            fy,cwtcohy = cwt_coherence_auto_nskip(v_[:,1],np.random.normal(size=len(v_)),
                                         sampling_period=1/sampling_rate,period_multiple=3)
            fz,cwtcohz = cwt_coherence_auto_nskip(v_[:,2],np.random.normal(size=len(v_)),
                                         sampling_period=1/sampling_rate,period_multiple=3)
            Cnullx.append( cwtcohx )
            Cnully.append( cwtcohy )
            Cnullz.append( cwtcohz )
        Cnullx = np.vstack(Cnullx)
        Cnully = np.vstack(Cnully)
        Cnullz = np.vstack(Cnullz)
        
        mux,stdx = Cnullx.mean(0),Cnullx.std(0)
        muy,stdy = Cnully.mean(0),Cnully.std(0)
        muz,stdz = Cnullz.mean(0),Cnullz.std(0)
        
        return fx,fy,fz,mux,muy,muz,stdx,stdy,stdz

    fx,fy,fz,cohmux,cohmuy,cohmuz,cohstdx,cohstdy,cohstdz = zip(*pool.map(f,t0))
    
    return ( (fx[0],np.vstack(cohmux),np.vstack(cohstdx)),
             (fy[0],np.vstack(cohmuy),np.vstack(cohstdy)),
             (fz[0],np.vstack(cohmuz),np.vstack(cohstdz)) )

def check_coherence_with_null(t0,subv,avv,tnull,cohnullmu,cohnullstd,
                              sampling_rate=30):
    """
    Given subject's trajectory compare it with the given null and return the fraction of
    frequencies at which the subject exceeds the null.
    
    Parameters
    ----------
    t0 : float
        Time relative to the start of the avatar velocity data at which subject and 
        avatar velocities arrays start.
    subv : ndarray
        Subject vel
    avv : ndarray
        Avatar vel
    tnull : ndarray
        Times at which null calculations were made.
    cohnullmu : ndarray
        Dimensions of (n_time,n_freq)
    cohnullstd : ndarray
    
    Returns
    -------
    performanceMetric
    """
    # Calculate coherence between given signals.
    f,cwtcoh = cwt_coherence_auto_nskip(subv,avv,
                                        sampling_period=1/sampling_rate,period_multiple=3)
    # Simple (not very good) check to make sure cwt was calculated in the same way.
    assert len(cwtcoh)==cohnullmu.shape[1],'%d, %d'%(len(cwtcoh),cohnullmu.shape[1])
    
    tIx = np.argmin(np.abs(tnull-t0))
    
    # Compare coherence will the sample of coherencen ulls given.
    # Ignore nans
    notnanix = (np.isnan(cwtcoh)|np.isnan(cohnullmu[tIx]))==0
    betterPerfFreqIx = cwtcoh[notnanix]>(cohnullmu[tIx][notnanix]+cohnullstd[tIx][notnanix]/2)
    
    return ( betterPerfFreqIx ).mean()

def phase_coherence(x,y):
    """
    Parameters
    ----------
    x : ndarray
    y : ndarray
    S : ndarray
        Smoothing filter for 2d convolution.

    Returns
    -------
    Phase coherence
    """
    xcwt,f = pywt.cwt(x,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)
    ycwt,f = pywt.cwt(y,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)

    smoothx = np.abs(xcwt)
    smoothy = np.abs(ycwt)
    smoothxy = xcwt*ycwt.conjugate()

    smoothcoh = smoothxy.mean(1) / ( smoothx*smoothy ).mean(1)
    return f,smoothcoh

def tf_phase_coherence(x,y,S):
    """
    Parameters
    ----------
    x : ndarray
    y : ndarray
    S : ndarray
        Smoothing filter for 2d convolution.

    Returns
    -------
    """
    from scipy.signal import convolve2d

    xcwt,f = pywt.cwt(x,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)
    ycwt,f = pywt.cwt(y,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)

    smoothx = convolve2d(np.abs(xcwt)**2,S,mode='same')
    smoothy = convolve2d(np.abs(ycwt)**2,S,mode='same')
    smoothxy = convolve2d(xcwt*ycwt.conjugate(),S,mode='same')
    
    smoothcoh = smoothxy.mean(1) / np.sqrt(( smoothx*smoothy ).mean(1))
    return f,smoothcoh

def tf_coherence(x,y,S):
    """
    Parameters
    ----------
    x : ndarray
    y : ndarray
    S : ndarray
        Smoothing filter for 2d convolution.
    """
    from scipy.signal import convolve2d

    xcwt,f = pywt.cwt(x,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)
    ycwt,f = pywt.cwt(y,np.logspace(0,2,100),'cgau1',sampling_period=1/60,precision=12)

    smoothx = convolve2d(np.abs(xcwt)**2,S,mode='same')
    smoothy = convolve2d(np.abs(ycwt)**2,S,mode='same')
    smoothxy = convolve2d(xcwt*ycwt.conjugate(),S,mode='same')

    smoothcoh = smoothxy/np.sqrt(smoothx*smoothy)
    return f,smoothcoh

def coherence_before_vis(subcwt,avcwt,f,vis,dt,min_freq=0,max_freq=10):
    """
    Coherence using the wavelet transform for dt seconds around the visibility turning back on.
    
    Parameters
    ----------
    subcwt
    avcwt
    f : ndarray
        Frequencies.
    vis
    dt : float
        temporal distance from the start of a new visibility section. Positive value is for
        before visibility starts.
    min_freq : float,0
    max_freq : float,10

    Returns
    -------
    Average coherence between (min_freq,max_freq).
    """
    # Get indices of points near the end of the invisibility window.
    dtprev = int(dt*60)
    visStartIx = np.where(np.diff(vis)==1)[0]-dtprev
    visStartIx = visStartIx[(visStartIx>=0)&(visStartIx<len(vis))]

    Psub = ( np.abs(subcwt[:,visStartIx])**2 ).mean(-1)
    Pav = ( np.abs(avcwt[:,visStartIx])**2 ).mean(-1)
    Pcross = ( subcwt[:,visStartIx]*avcwt[:,visStartIx].conjugate() ).mean(-1)
    
    coh = np.abs(Pcross)**2/Psub/Pav
    freqix = (f>=min_freq)&(f<=max_freq)

    # Errors.
    #print ( Psub/( np.abs(subcwt[:,visStartIx])**2 ).std(-1) )[freqix]
    #print ( Pav/( np.abs(avcwt[:,visStartIx])**2 ).std(-1) )[freqix]
    #print ( np.abs(Pcross)**2/(np.abs( subcwt[:,visStartIx]*avcwt[:,visStartIx].conjugate()
    #    )**2).std(-1) )[freqix]

    avgC = np.trapz(coh[freqix],x=f[freqix]) / (f[freqix].max()-f[freqix].min())
    if avgC<0:
        return -avgC
    return avgC

def cwt_coherence(x,y,nskip,
                  scale=np.logspace(0,2,100),
                  sampling_period=1/60,
                  **kwargs):
    """
    Use the continuous wavelet transform to measure coherence.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    nskip : int
        Number of indices to skip when averaging across spectra for coherence. This is to reduce
        correlation between samples when averaging.
    scale : list
        Scale of continuous wavelets.
    sampling_period : float,1/60
        Used to choose scales.
    **kwargs
        for pywt.cwt()

    Returns
    -------
    f : ndarray
    coherence : ndarray
    """
    import pywt
    assert len(x)==len(y)
    xcwt,f = pywt.cwt(x,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    ycwt,f = pywt.cwt(y,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    
    # Get indices of points with some overlap.
    selectix = np.arange(nskip,len(x),nskip,dtype=int)

    Psub = ( np.abs(xcwt[:,selectix])**2 ).mean(-1)
    Pav = ( np.abs(ycwt[:,selectix])**2 ).mean(-1)
    Pcross = ( xcwt[:,selectix]*ycwt[:,selectix].conjugate() ).mean(-1)
    coh = np.abs(Pcross)**2/Psub/Pav
    
    # Skip low frequencies that have periods longer the duration of the window.
    fCutoff = f<(1/(len(x)*sampling_period))

    return f[f>=fCutoff],coh[f>=fCutoff]

def cwt_coherence_auto_nskip(x,y,
                             scale=np.logspace(0,2,100),
                             sampling_period=1/60,
                             period_multiple=1,
                             **kwargs):
    """
    Use the continuous wavelet transform to measure coherence but automatically choose the amount to
    subsample separately for each frequency when averaging. The subsampling is determined by nskip
    which only takes a sample every period of the relevant frequency.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    nskip : int
        Number of indices to skip when averaging across spectra for coherence. This is to reduce
        correlation between samples when averaging.
    scale : list
        Scale of continuous wavelets.
    sampling_period : float,1/60
        Used to choose scales.
    **kwargs
        for pywt.cwt()

    Returns
    -------
    f : ndarray
    coherence : ndarray
    """
    import pywt
    assert len(x)==len(y)
    xcwt,f = pywt.cwt(x,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    ycwt,f = pywt.cwt(y,scale,'cgau1',sampling_period=sampling_period,**kwargs)
    
    Psub = np.zeros(len(f))
    Pav = np.zeros(len(f))
    Pcross = np.zeros(len(f),dtype=complex)
    #PsubStd = np.zeros(len(f))
    #PavStd = np.zeros(len(f))
    #PcrossStd = np.zeros(len(f))

    # For each freq, skip roughly by a period.
    for fIx,f_ in enumerate(f):
        nskip = int(1/f_/sampling_period)
        if nskip>(len(x)//3):
            Psub[fIx] = np.nan
            Pav[fIx] = np.nan
            Pcross[fIx] = np.nan
        else:
            selectix = np.arange(nskip,len(x),nskip,dtype=int)
            #print f_,len(selectix)

            Psub[fIx] = ( np.abs(xcwt[fIx,selectix])**2 ).mean(-1)
            Pav[fIx] = ( np.abs(ycwt[fIx,selectix])**2 ).mean(-1)
            Pcross[fIx] = ( xcwt[fIx,selectix]*ycwt[fIx,selectix].conjugate() ).mean(-1)

            #PsubStd[fIx] = ( np.abs(xcwt[fIx,selectix])**2 ).std(-1)
            #PavStd[fIx] = ( np.abs(ycwt[fIx,selectix])**2 ).std(-1)
            #PcrossStd[fIx] = ( xcwt[fIx,selectix]*ycwt[fIx,selectix].conjugate() ).std(-1)

    coh = np.abs(Pcross)**2/Psub/Pav
    #stds = (PsubStd,PavStd,PcrossStd)

    # Skip low frequencies that have periods longer the duration of the window.
    fCutoffIx = f>(period_multiple/(len(x)*sampling_period))

    return f[fCutoffIx],coh[fCutoffIx]

def max_coh_time_shift(subv,temv,
                       dtgrid=np.linspace(0,1,100),
                       mx_freq=10,
                       sampling_rate=60,
                       window_width=2,
                       disp=False,
                       ax=None):
    """
    Find the global time shift that maximizes the coherence between two signals.
    
    Parameters
    ----------
    subv : ndarray
        Subject time series. If multiple cols, each col is taken to be a data point and the average
        coherence is maximized.
    temv : ndarray
        Template time series.
    dtgrid : ndarray,np.linspace(0,1,100)
    window_width : float,2
        Window duration for computing coherence in terms of seconds.
    disp : bool,False
    ax : AxesSubplot,None
        
    Returns
    -------
    dt : float
        Time shift in seconds that maximizes scipy coherence. Time shift is relative to subject
        time, i.e.  negative shift is shifting subject back in time and positive is shifting subject
        forwards in time. If subject is tracking template, then dt>0.
    maxcoh : float
        Coherence max.
    """
    from scipy.signal import coherence
        
    # Convert dtgrid to index shifts.
    dtgrid = np.unique(np.around(dtgrid*sampling_rate).astype(int))
    if subv.ndim==1:
        coh = np.zeros(len(dtgrid))
    else:
        coh = np.zeros((len(dtgrid),subv.shape[1]))
    window_width = int(sampling_rate*window_width)
    
    def _calc_coh(subv,temv):
        for i,dt in enumerate(dtgrid):
            if dt<0:
                f,c = coherence(subv[-dt:],temv[:dt],fs=sampling_rate,nperseg=window_width)
            elif dt>0:
                f,c = coherence(subv[:-dt],temv[dt:],fs=sampling_rate,nperseg=window_width)
            else:
                f,c = coherence(subv,temv,fs=sampling_rate,nperseg=window_width)
            coh[i] = abs(c)[f<mx_freq].mean()
        return coh
        
    if subv.ndim==1:
        coh = _calc_coh(subv,temv)
    else:
        coh = np.vstack([_calc_coh(subv[:,i],temv[:,i]) for i in xrange(subv.shape[1])]).mean(1)
    shiftix = np.argmax(coh)
    
    if disp:
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(dtgrid/sampling_rate,coh,'o')
        ax.set(xlabel='dt',ylabel='coherence')
        
    return dtgrid[shiftix]/sampling_rate,coh[shiftix]



# ======= #
# Classes #
# ======= #
class DTWPerformance(object):
    def __init__(self,dwt_kwargs={'dist':2}):
        """
        Class for using fast DWT to compare two temporal trajectories and return a performance metric.
        
        Parameters
        ----------
        """
        self.dwtSettings = dwt_kwargs

    def compare(self,x,y,dt=1.):
        """
        Parameters
        ----------
        x : ndarray
        y : ndarray
        dt : float,1
            Sampling rate for x and y.

        Returns
        -------
        inner : float
            Normalized average inner product between x and y when indexed by warped path.
        dtmax : float
            Largest timing different from DTW algorithm.
        """
        from numpy.linalg import norm

        dist,path = fastdtw(x,y,**self.dwtSettings)
        path = np.vstack(path)

        # Calculate correlation between the two vectors.
        inner = ( (x[path[:,0]]*y[path[:,1]]).sum(1) / 
                  (norm(x[path[:,0]],axis=1)*norm(y[path[:,1]],axis=1)) ).mean()
        dtmax = np.abs(np.diff(path,axis=1)).max() * dt

        return inner,dtmax

class CoherenceEvaluator(object):
    '''
    update() evaluates the average coherence over the given time.
    These assume V_person and V_avatar are pre-aligned and have the same length.
    '''
    def __init__(self,maxfreq,sample_freq=60,window_length=90):
        '''
        Parameters
        ----------
        maxfreq : float
            Max frequency up to which to average coherence.
        sampleFreq : float,60
            Sampling frequency
        windowLength : int,90
            Number ofdata points to use in window for coherence calculation.

        Subfields
        ---------
        coherence
        coherences
        '''
        self.maxfreq = maxfreq
        self.sampleFreq = sample_freq
        self.windowLength = window_length
        
        self.v = None
        self.v_av = None
        
        self.coherence = 0
        self.coherences = np.empty(0)
        self.performanceValues = np.empty(0)
        
    def getCoherence(self):
        return self.coherence
    
    def getAverageCoherence(self):
        '''
        For GPR: returns average coherence over a full trial. Coherences should
        then be reset for the new trial.
        '''
        return np.mean(self.coherences)
    
    def resetCoherences(self):
        self.coherences = np.empty(0)
    
    def getPerformanceValues(self):
        return self.performanceValues
    
    def getAveragePerformance(self):
        return np.mean(self.performanceValues)
    
    def resetPerformance(self):
        self.performanceValues = np.empty(0)
        
    def evaluateCoherence(self,v1,v2,use_cwt=True):
        '''
        Returns average coherence between current v and v_av data vectors.

        Parameters
        ----------
        v1 : ndarray
            Vector.
        v2 : ndarray
            Vector.

        Returns
        -------
        avg_coh : float
        '''
        assert len(v1)==len(v2)
            
        if not use_cwt:
            # Scipy.signal's coherence implementation.
            self.f,self.C = coherence(v1,v2,
                                      fs=self.sampleFreq,
                                      nperseg=self.windowLength,
                                      nfft=2 * self.windowLength,
                                      noverlap=self.windowLength//4)
        else:
            self.f,self.C = cwt_coherence(v1,v2,1,sampling_period=1/self.sampleFreq)
            self.C *= -1 

        # Evaluate Average Coherence by the Trapezoid rule.
        freqIx = (self.f>0)&(self.f<self.maxfreq)
        avg_coherence = np.trapz(self.C[freqIx],x=self.f[freqIx]) / (self.f[freqIx].max()-self.f[freqIx].min())
        
        if np.isnan(avg_coherence): avg_coherence = 0.
        return avg_coherence
    
    def evaluatePerformance(self):
        '''
        Evaluates average coherence against threshold value, and writes binary
        value to target output file.

        Returns
        -------
        performance
        '''
        performance = 0
        avg_coherence = self.evaluateCoherence()
        
        if avg_coherence >= self.performanceThreshold:
            performance = 1
        
        self.coherences = np.append(self.coherences,avg_coherence)
        self.performanceValues = np.append(self.performanceValues,performance)
        return performance
# end CoherenceEvaluator



class GPR(object):
    '''
    Class performs the gpr and writes the new visibility fraction/time to outputFile
    '''
    def __init__(self,
                 GPRKernel = 1.0 * RBF(length_scale=np.array([.5,.2]), length_scale_bounds=(1e-1, 10.0)),
                 tmin=0.5,tmax=2,tstep=0.1,
                 fmin=0.1,fmax=0.9,fstep=0.1):
        '''
        Parameters
        ----------
        tmin : float
            minimum window time
        fmin : float
            minimum visibility fraction.
        '''
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.fmin = fmin
        self.fmax = fmax
        self.fstep = fstep
        
        self.kernel = GPRKernel
        
        self.durations = np.zeros(0)
        self.fractions = np.zeros(0)
        self.coherences = np.zeros(0)
        
        # Create two grids for t and f.
        self.meshPoints = np.meshgrid(np.arange(self.tmin,self.tmax+self.tstep,self.tstep),
                                      np.arange(self.fmin,self.fmax+self.fstep,self.fstep))
        # Flatten t and f grids and stack them into an Nx2 array.
        self.meshPoints = np.vstack([x.ravel() for x in self.meshPoints]).T
        
        self.gp = gaussian_process.GaussianProcessRegressor(kernel=self.kernel)
        self.coherence_pred = 0
        self.std_pred = 0
   
    def predict(self):
        '''
        Calls the GPR
        '''
        self.gp.fit( np.vstack((self.durations,self.fractions)).T,self.coherences )
        self.coherence_pred, self.std_pred = self.gp.predict(self.meshPoints,return_std=True)
        
    def max_uncertainty(self):
        '''
        Returns next_duration,next_fraction as the point where the variance of the GPR is max
        Currently finds maximum uncertainty, and then returns a point with that uncertainty as the
        update value. But really this should be a point which would minimize the total uncertainty.
        
        Returns
        -------
        next_window_duration : float
        next_vis_fraction : float
        '''
        maxIndex = np.argmax(self.std_pred)

        next_duration = self.meshPoints[maxIndex][0]
        next_fraction = self.meshPoints[maxIndex][1]
        
        return next_duration,next_fraction
        
    def update(self,new_coherence,window_dur,vis_fraction):
        '''
        This is called to add new data point to prediction.

        Parameters
        ----------
        new_coherence : float
        window_dur : float
        vis_fraction : float
        '''
        self.coherences = np.append(self.coherences,new_coherence)
        self.fractions = np.append(self.fractions,vis_fraction)
        self.durations = np.append(self.durations,window_dur)
        
        self.predict()
        
        return self.max_uncertainty()
#end GPR
