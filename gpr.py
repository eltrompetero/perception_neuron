from __future__ import division
import numpy as np
from scipy.signal import coherence
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF


class CoherenceEvaluator(object):
    '''
    update() evaluates the average coherence over the given time.
    These assume V_person and V_avatar are pre-aligned and have the same length.
    '''
    def __init__(self,performanceThreshold,maxfreq,
                 SAMPLE_FREQ,WINDOW_LENGTH):
        '''
        Window length in seconds.
        performanceThreshold, a number between 0 and 1, specifies the least that a
        person's average coherence should be for a performance grade of 1 (good).

        Parameters
        ----------
        performanceThreshold : float
        maxfreq : float
            Max frequency up to which to average coherence.
        SAMPLE_FREQ : float
        WINDOW_LENGTH : int
        '''
        self.maxfreq = maxfreq
        self.SAMPLE_FREQ = SAMPLE_FREQ
        self.WINDOW_LENGTH = WINDOW_LENGTH
        
        self.performanceThreshold = performanceThreshold
        
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
        
    def evaluateCoherence(self,v1,v2):
        '''
        Returns average coherence between current v and v_av data vectors.
        '''
        # Scipy.signal's coherence implementation. Not yet Eddie-coherence.
        self.f,self.C = coherence(v1,v2,
                                  fs=self.SAMPLE_FREQ,
                                  nperseg=self.WINDOW_LENGTH,
                                  nfft=2 * self.WINDOW_LENGTH)
        
        # Evaluate Average Coherence by the Trapezoid rule.
        freqIx = (self.f>0)&(self.f<self.maxfreq)
        avg_coherence = np.trapz(self.C[freqIx],x=self.f[freqIx]) / (self.f[freqIx].max()-self.f[freqIx].min())
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
                 GPRKernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                 TMIN=0.5,TMAX=2,TSTEP=0.1,
                 FMIN=0.1,FMAX=0.9,FSTEP=0.1):
        '''
        TMIN is minimum window time; FMIN minimum visibility fraction.
        '''
        self.TMIN = TMIN
        self.TMAX = TMAX
        self.TSTEP = TSTEP
        self.FMIN = FMIN
        self.FMAX = FMAX
        self.FSTEP = FSTEP
        
        self.kernel = GPRKernel
        
        self.durations = np.zeros(0)
        self.fractions = np.zeros(0)
        self.coherences = np.zeros(0)
        
        self.meshPoints = np.meshgrid(np.arange(self.TMIN,self.TMAX+self.TSTEP,self.TSTEP),
                                      np.arange(self.FMIN,self.FMAX+self.FSTEP,self.FSTEP))
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
        # Currently finds maximum uncertainty,and then returns a point with
        # that uncertainty as the update value. But really this should be a
        # point which would minimize the total uncertainty.

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
