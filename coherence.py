# ================================================================================================ # 
# Functions for calculating performance measures.
# Authors: Eddie Lee, edl56@cornell.edu
#          Ted Esposito
# ================================================================================================ # 

from __future__ import division
import numpy as np
from scipy.signal import coherence
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from warnings import warn



def precompute_coherence_nulls(v,t0,windowDuration,pool,
        sampling_rate=30,n_iters=100):
    """
    This is unnecessary! As noted in Tango NBIII 2017-10-30.
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

def check_coherence_with_null(ref,sample,threshold,
                              sampling_rate=30):
    """
    Given subject's trajectory compare it with the given null and return the fraction of
    frequencies at which the subject exceeds the white noise null which is just a flat cutoff.
    
    Parameters
    ----------
    ref : ndarray
        Reference signal against which to compare the sample. This determines the noise
        threshold.
    sample : ndarray
        Sample signal to compare against reference signal.
    threshold : float
        This determines the constant with which to multiply the power spectrum of the reference
        signal to determine the null cutoff.

    Returns
    -------
    performanceMetric : float
    """
    assert 0<=threshold<=1
    
    # Calculate coherence between given signals.
    f,cwtcoh = cwt_coherence_auto_nskip(ref,sample,
                                        sampling_period=1/sampling_rate,
                                        period_multiple=3)
    
    # Ignore nans
    notnanix = np.isnan(cwtcoh)==0
    betterPerfFreqIx = cwtcoh[notnanix]>threshold
    
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
    def __init__(self,inner_prod_threshold=.3,
                 norm_dv_threshold=.1,
                 norm_dv_ratio_threshold=np.log2(1.5),
                 dt_threshold=.5,
                 dwt_kwargs={'dist':2}):
        """
        Class for using fast DWT to compare two temporal trajectories and return a performance metric.
        Performance is the fraction of time warped trajectories that the individuals remain within some
        threshold preset.
        
        Parameters
        ----------
        inner_prod_threshold : float,.5
            Max deviation allowed for one minus normalized dot product between vectors.
        norm_dv_threshold : float,.1
            Max difference in speeds allowed. Units of m/s.
        norm_dv_ratio_threshold : float,1
            Max ratio in speeds allowed in units of log2.
        dt_threshold : float,.5
            Max deviation for time allowed.
        """
        self.dwtSettings = dwt_kwargs
        self.innerThreshold = inner_prod_threshold
        self.normdvThreshold = norm_dv_threshold
        self.normdvRatioThreshold = norm_dv_ratio_threshold
        self.dtThreshold = dt_threshold

    def time_average(self,x,y,dt=1.,strict=False):
        """
        Measure performance as the fraction of time you are within the thresholds.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        dt : float,1
            Sampling rate for x and y.

        Returns
        -------
        performance : float
            Fraction of the length of given trajectories that are within set thresholds.
        """
        from numpy.linalg import norm
        from fastdtw import fastdtw

        dist,path = fastdtw(x,y,**self.dwtSettings)
        path = np.vstack(path)

        normx = norm(x[path[:,0]],axis=1)+np.nextafter(0,1)
        normy = norm(y[path[:,1]],axis=1)+np.nextafter(0,1)
        # Dot product between the two vectors.
        inner = (x[path[:,0]]*y[path[:,1]]).sum(1) / normx / normy
        # Relative norms.
        normDiff = np.abs(normx-normy)
        normRatio = np.abs(np.log2(normx)-np.log2(normy))
        dt = np.diff(path,axis=1) * dt

        # Calculate performance metric.
        # In strict case, dt must be within cutoff at all times to get a nonzero performance value.
        # Otherwise, we just take the average time during which subject is within all three norm, inner
        # product, and dt cutoffs.
        if strict:
            if (np.abs(dt)<self.dtThreshold).all():
                performance = ((1-inner)<self.innerThreshold).mean()
            else:
                performance = 0.
        else:
            performance = ( #((normDiff<self.normdvThreshold)|(normRatio<self.normdvRatioThreshold)) &
                            ((1-inner)<self.innerThreshold) & 
                            (np.abs(dt)<self.dtThreshold) ).mean()
        
        return performance

    def raw(self,x,y,dt=1.):
        """
        Performance as measured by the similarity of time warped trajectories. If time warping is too big,
        then performance is 0.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        dt : float,1
            Sampling rate for x and y.

        Returns
        -------
        performance : float
        """
        from numpy.linalg import norm
        from fastdtw import fastdtw

        dist,path = fastdtw(x,y,**self.dwtSettings)
        path = np.vstack(path)

        # Calculate dot product between the two vectors.
        inner = ( (x[path[:,0]]*y[path[:,1]]).sum(1) / 
                  (norm(x[path[:,0]],axis=1)*norm(y[path[:,1]],axis=1)+np.nextafter(0,1)) )
        dt = np.diff(path,axis=1) * dt

        # Calculate performance metric.
        if (np.abs(dt)<self.dtThreshold).all():
            performance = inner.mean()
            if performance<0:
                performance = 0
        else:
            performance = 0.
        return performance
#end DTWPerformance



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
    def __init__(self,
                 GPRKernel = RBF(length_scale=np.array([1.,1.])),
                 alpha = .2,
                 mean_performance=np.log(1),
                 length_scale=np.array([1.,.2]),
                 tmin=0.5,tmax=2,tstep=0.1,
                 fmin=0.1,fmax=1.,fstep=0.1):
        '''
        Wrapper around GPR class to perform useful functions for HandSyncExperiment.

        Parameters
        ----------
        GPRKernel : sklearn.gaussian_processes.kernels.RBF
        alpha : float
            Uncertainty in diagonal matrix for GPR kernel.
        mean_performance : float,.5
            By default, the sigmoid is centered around 0, the mean of the Gaussian process, corresponding to
            perf=0.5. However, we should center the sigmoid around the mean value of y which is specified
            here. Since the GPR is trained in the logistic space, the offset is given by the logistic offset.
            The mean is automatically accounted for under the hood, so you don't have to worry about adding or
            subtracting it in the interface.
        length_scale : ndarray
            (duration_scale,fraction_scale) Remember that duration is the angle about the origin restricted to
            be between [0,pi] and fraction is the radius. The GPR learns once (r,theta) has been mapped to the
            Cartesian coordinate system.
        tmin : float,0.5
            minimum window time
        tmax : float,2
        tstep : float,0.1
        fmin : float
            minimum visibility fraction.
        fmax : float
        fstep : float

        Members
        -------
        fractions : ndarray
            Fraction of time stimulus is visible.
        durations : ndarray
            Duration of window.
        meshPoints : ndarray
            List of grid points (duration,fraction) over which performance was measured.
        '''
        assert (type(length_scale) is np.ndarray) and len(length_scale)==2

        from gaussian_process.regressor import GaussianProcessRegressor
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.fmin = fmin
        self.fmax = fmax
        self.fstep = fstep
        
        self.length_scale = length_scale
        self.kernel = self.handsync_experiment_kernel(length_scale)
        self.alpha = alpha
        
        self.durations = np.zeros(0)
        self.fractions = np.zeros(0)
        self.performanceData = np.zeros(0)
        self.mean_performance = mean_performance
        
        # Create two grids for t and f.
        self.tRange = np.arange(self.tmin,self.tmax+self.tstep,self.tstep)
        self.fRange = np.arange(self.fmin,self.fmax+self.fstep,self.fstep)
        self.meshPoints = np.meshgrid(self.tRange,self.fRange)
        # Flatten t and f grids and stack them into an Nx2 array.
        self.meshPoints = np.vstack([x.ravel() for x in self.meshPoints]).T
        
        self.gp = GaussianProcessRegressor(self.kernel,alpha**-2)
        self.performanceGrid = 0  # [-inf,inf]
        self.std_pred = 0

        self.pointsToAvoid = []
   
    def predict(self,mesh=None):
        '''
        Fits the GPR to all data points and saves the predicted values with errors. The mean in the target
        perf values is accounted for here.

        Parameters
        ----------
        mesh : ndarray
            Points at which to evaluate GPR. Should be (samples,2) with first column durations and the second
            fraction of visible window.

        Returns
        -------
        perf : ndarray
            Performance grid.
        perfErr : ndarray
            Performance estimated standard deviation.
        '''
        if mesh is None:
            mesh = self.meshPoints

        self.gp.fit( np.vstack((self.durations,self.fractions)).T,self.performanceData-self.mean_performance )
        self.performanceGrid, self.std_pred = self.gp.predict(mesh,return_std=True)
        self.performanceGrid += self.mean_performance

        return self.performanceGrid.copy(),self.std_pred.copy()

    def grad(self,eps=1e-5):
        '''
        Estimates the gradient at each point of the mesh.

        Parameters
        ----------
        eps : float,1e-5

        Returns
        -------
        grad : ndarray
            Dimensions (n_tRange,n_fRange,2). Last dimension corresponds to the gradient along each diemnsion
            of the input.
        '''
        grad = np.zeros((len(self.meshPoints),2))
        X1 = self.meshPoints.copy()
        X0 = self.meshPoints.copy()

        X1[:,0] += eps 
        X0[:,0] -= eps 
        grad[:,0] = ( self.gp.predict(X1)-self.gp.predict(X0) )/(2*eps)
        X1[:,0] -= eps
        X0[:,0] += eps

        X1[:,1] += eps
        X0[:,1] -= eps
        grad[:,1] = ( self.gp.predict(X1)-self.gp.predict(X0) )/(2*eps)
        
        shape = len(self.fRange),len(self.tRange)
        grad = np.concatenate((grad[:,0].reshape(shape)[:,:,None],grad[:,1].reshape(shape)[:,:,None]),2)
        return grad
        
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
        
    def select_contour(self,pstar,choose_algo='',algo_kwargs={}):
        """
        Select the point in the field with mean closest to desired performance value in the [0,1] space.

        Option to avoid points so that we don't pick the same points over and over again.

        Parameters
        ----------
        pstar : ndarray
            Performance value around which to choose points. This is in the [-inf,inf] stretched space.
        choose_algo : str,''
            If 'avoid', do not sample points in self.pointsToAvoid. Typically, this will be a stored list of
            points already sampled.
            If 'err', weight choice by the uncertainty. Choose the point that minimize the distance to pstar
            with max error. First, thresholds points by distance to pstar. Must specify 'threshold' and
            'std_scale'.
        algo_kwargs : dict,{}
            Any keyword args needed by algorithm for choosing the next point.

        Returns
        -------
        duration : float
        fraction : float
        """
        if type(pstar) is int:
            pstar = float(pstar)
        if type(pstar) is float:
            pstar = np.array([pstar])
        
        if choose_algo=='avoid':
            ix = []
            for pstar_ in pstar:
                sortix = np.argsort( np.abs(self.logistic(self.performanceGrid)-pstar_) )
                counter = 0
                while counter<len(sortix):
                    if not any(np.array_equal(self.meshPoints[sortix[counter]],x)
                               for x in self.pointsToAvoid):
                        ix.append(sortix[counter])
                        counter = len(sortix)
                    counter += 1
        elif choose_algo=='err':
            assert not algo_kwargs.get('threshold',None) is None
            assert not algo_kwargs.get('std_scale',None) is None

            ix = []
            for pstar_ in pstar:
                dist = np.abs(self.logistic(self.performanceGrid)-pstar_)
                thresholdIx = dist<algo_kwargs['threshold']
                dist[thresholdIx==0] += 1e30
                ix.append( np.argmin(dist-algo_kwargs['std_scale']*self.std_pred) )
        else:
            ix = [np.argmin(np.abs(self.performanceGrid-pstar_)) for pstar_ in pstar]
        return self.meshPoints[ix,0],self.meshPoints[ix,1]
    
    @staticmethod
    def _scale_erf(x,mu,std):
        from scipy.special import erf
        return .5 + erf( (x-mu)/std/np.sqrt(2) )/std/2

    def update(self,new_performance,window_dur,vis_fraction):
        '''
        This is called to add new data point to prediction.

        Parameters
        ----------
        new_performance : float
        window_dur : float
        vis_fraction : float
        '''
        self.performanceData = np.append(self.performanceData,new_performance)
        self.fractions = np.append(self.fractions,vis_fraction)
        self.durations = np.append(self.durations,window_dur)
        
        self.predict()
    
    @staticmethod
    def ilogistic(x):
        return -np.log(1/x-1)
    
    @staticmethod
    def logistic(x):
        return 1/(1+np.exp(-x))

    def handsync_experiment_kernel(self,length_scales):
        """
        Calculates the RBF kernel for one pair of (window_duration,visible_fraction). Specifically for the
        handsync experiment.
        
        Custom kernel converts (duration,fraction) radial coordinate system pair into Cartesian coordinates
        and then calculates distance to calculate GPR in this space.

        Parameters
        ----------
        length_scales : ndarray
            2 element vector specifying (duration,fraction)

        Returns
        -------
        kernel : function
        """
        from numpy.linalg import norm
        assert length_scales[0]>=(self.tmax/np.pi)
        delta = define_delta(1,width=.0)
        
        def kernel(tf0,tf1,length_scales=length_scales):
            xy0 = np.array([ (1-tf0[1])/length_scales[1]*np.cos(tf0[0]/length_scales[0]),
                             (1-tf0[1])/length_scales[1]*np.sin(tf0[0]/length_scales[0]) ])
            xy1 = np.array([ (1-tf1[1])/length_scales[1]*np.cos(tf1[0]/length_scales[0]),
                             (1-tf1[1])/length_scales[1]*np.sin(tf1[0]/length_scales[0]) ])

            return np.exp( -((xy0-xy1)**2).sum() )
        return kernel
#end GPR



def define_delta(x,width=0.):
    """
    Return delta function or Gaussian approximation with given width.

    Parameters
    ----------
    x : float
        delta(x-y)
    width : float,0.

    Returns
    -------
    function
    """
    if width>0:
        return lambda xp:np.exp(-(xp-x)**2/2/width**2)
    return lambda xp:0. if xp!=x else 1.

def handsync_experiment_kernel(self,length_scales):
        """
        For backwards compatibility. See GPR. 
        """
        warn("Only for backwards compatibility.")
        from numpy.linalg import norm
        delta = define_delta(1,width=.00)
        
        def kernel(tf0,tf1,length_scales=length_scales):
            xy0 = np.array([ (1-tf0[1])/length_scales[1]*np.cos(tf0[0]/length_scales[0]),
                             (1-tf0[1])/length_scales[1]*np.sin(tf0[0]/length_scales[0]) ])
            xy1 = np.array([ (1-tf1[1])/length_scales[1]*np.cos(tf1[0]/length_scales[0]),
                             (1-tf1[1])/length_scales[1]*np.sin(tf1[0]/length_scales[0]) ])

            return np.exp( -((xy0-xy1)**2).sum() )
        return kernel

