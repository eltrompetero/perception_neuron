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
    def __init__(self,coherenceEvaluator,
                 initialTime,initialVisibilityFraction,outputFile,
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
        self.outputFile = outputFile
        self.coherenceEvaluator = coherenceEvaluator
        
        self.kernel = GPRKernel
        
        self.times = np.empty(1)
        self.times[0] = initialTime
        self.fractions = np.empty(1)
        self.fractions[0] = initialVisibilityFraction
        self.coherences = np.empty(0)
        
        self.meshPoints = self.make_mesh(self.TMIN,self.TMAX,self.TSTEP,
                                         self.FMIN,self.FMAX,self.FSTEP)
        
        self.gp = gaussian_process.GaussianProcessRegressor(kernel=self.kernel)
        self.coherence_pred = 0
        self.variance_pred = 0
        
    def make_mesh(self,min_time,max_time,time_step,min_frac,max_frac,frac_step):
        '''Generates mesh points for multi-dimensional GPR'''
        time_range = np.linspace(min_time,max_time,time_step)
        frac_range = np.linspace(min_frac,max_frac,frac_step)
        
        t_length = len(time_range)
        f_length = len(frac_range)
        
        mesh_points = np.ndarray((t_length*f_length,2))

        for i in range(t_length):
            for j in range(f_length):
                mesh_points[(i+1) + j][0] = time_range[i]
                mesh_points[(i+1) + j][1] = frac_range[j]
        
        return mesh_points
    
    def getDataPoints(self):
        '''Converts N input time,fraction pairs into an Nx2 matrix '''
        data_points = np.ndarray((len(self.times),2))
        for i in range(len(self.times)):
            data_points[i][0] = self.times[i]
            data_points[i][1] = self.fractions[i]
            
        return data_points
    
    def predict(self):
        '''
        Calls the GPR
        '''
        dataPoints = self.getDataPoints()
            
        self.gp.fit(dataPoints,self.coherences)
        
        self.coherence_pred, self.variance_pred = self.gp.predict(self.meshPoints)
        
    def max_uncertainty(self):
        '''
        Returns next_time,next_fraction as the point where the variance of the GPR is max
        '''
        # Currently finds maximum uncertainty,and then returns a point with
        # that uncertainty as the update value. But really this should be a
        # point which would minimize the total uncertainty.
        
        next_time = 0
        next_fraction = 0
        maxIndex = 0
        max_var = max(self.variance_pred)
        
        for i in range(len(self.variance_pred[0])):
            if self.variance_pred[i] == max_var:
                maxIndex = i
                break
        
        next_time = self.meshPoints[maxIndex][0]
        next_fraction = self.meshPoints[maxIndex][1]
        
        return (next_time,next_fraction)
        
    def update(self):
        '''
        This is called to write a new time/fraction pair to outputFile
        Output: next_time,next_fraction
        '''
        new_coherence = self.coherenceEvaluator.getAverageCoherence()
        self.coherenceEvaluator.resetCoherences()
        self.coherences = np.append(self.coherences,new_coherence)
        
        self.predict()
        
        (next_time,next_fraction) = self.max_uncertainty()
        next_fraction = self.max_uncertainty()
        self.nextQueryOutput = open(self.outputFile,'w')
        self.nextQueryOutput.write(str(next_time) + ',' + str(next_fraction))
        self.nextQueryOutput.write(str(next_fraction))
        self.nextQueryOutput.close()
        
        self.times = np.append(self.times,next_time)
        self.fractions = np.append(self.fractions,next_fraction)
        
