# ================================================================================================ # 
# Helper functions and modules that are imported into every other module in this package. These
# should not depend on other modules.
# 
# Author: Eddie Lee, edl56@cornell.edu
# 2017-03-28
# ================================================================================================ # 

from __future__ import division
try:
    import matplotlib.pyplot as plt
except ImportError:
    print "Could not import matplotlib."
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin,cos,pi
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline,interp1d
import entropy as info
from scipy.signal import fftconvolve
from misc.utils import unique_rows
from numba import jit
import multiprocess as mp
from scipy.optimize import minimize
from scipy.signal import spectrogram,savgol_filter,fftconvolve
from misc.angle import *
import pywt
import os

# Declare global variables for entire package.
DATADR = os.path.expanduser('~')+'/Dropbox/Sync_trials/Data'


# ================== # 
# Class definitions.
# ================== # 
class MultiUnivariateSpline(object):
    def __init__(self,t,x,knot_spacing=1/30,fit_type='LSQ',**kwargs):
        """
        Simple extension of UnivariateSpline to handle multidimensional data.
        
        Parameters
        ----------
        t (ndarray)
        x (ndarray)
            If 2d, then (n_samples,n_dim).
        knot_spacing : float,1/30
        fit_type : str,'LSQ'

        **kwargs
        """
        if type(t) is list:
            t = np.array(t)
        if type(x) is list:
            x = np.array(x)
        assert x.ndim==2
        self.splines = []
        self.knot_spacing = knot_spacing
        
        if fit_type=='LSQ':
            for i in xrange(x.shape[1]):
                self.splines.append(LSQUnivariateSpline(t,x[:,i],
                                        np.linspace(t[1],t[-2],int(np.floor(t[-2]-t[1])/knot_spacing)),
                                        **kwargs))
        elif fit_type=='Uni':
            for i in xrange(x.shape[1]):
                self.splines.append(UnivariateSpline(t,x[:,i],**kwargs))
        else:
            raise NotImplementedError

    def __call__(self,t):
        xSpline = np.zeros((len(t),len(self.splines)))
        for i in xrange(len(self.splines)):
            xSpline[:,i] = self.splines[i](t)
        return xSpline

    def derivative(self):
        self.dsplines = []
        for i in xrange(len(self.splines)):
            self.dsplines.append(self.splines[i].derivative())
        
        def f(t):
            xSpline = np.zeros((len(t),len(self.splines)))
            for i in xrange(len(self.splines)):
                xSpline[:,i] = self.dsplines[i](t)
            return xSpline
        return f



# ===================== #
# Function definitions. #
# ===================== #
def compare_specs(x,y,precision):
    """Given either one spec and a list or two lists, compare the twople values with each other to a
    precision that is a multiple of precision.

    Parameters
    ----------
    x : list of twoples or twople
    y : list of twoples or twople

    Returns
    -------
    comparison_result : ndarray
    """
    assert precision>=0

    # Case with both twoples.
    if type(x) is tuple and type(y) is tuple:
        if precision==0:
            if x[0]==y[0] and x[1]==y[1]:
                return True
            return False

        if abs(x[0]-y[0])<=precision and abs(x[1]-y[1])<=precision:
            return True
        return False

    # Case with one twople and one list.
    if type(x) is tuple and type(y) is list or (type(x) is list and type(y) is tuple):
        if type(x) is list:  # swap
            x_=x
            x=y
            y=x_

        comparison=np.zeros(len(y),dtype=bool)
        for i in xrange(len(y)):
            comparison[i]=compare_specs(x,y[i],precision)

    # Case with both lists.
    elif type(x) is list and type(y) is list:
        assert len(x)==len(y)
        comparison=np.zeros(len(y),dtype=bool)
        for i in xrange(len(y)):
            comparison[i]=compare_specs(x[i],y[i],precision)
    
    else:
        raise Exception("Unrecognized input.")

    return comparison

def _equiv_round(x,y,precision):
    return round(x/precision)==round(y/precision)

def rotate_xy(xy,theta):
    """
    Apply rotation to xy-vector.

    Parameters
    ----------
    xy : ndarray
        (n_samples,2)
    theta : float
        Angle to rotate by in the CCW direction.

    Returns
    -------

    """
    return np.array([[np.cos(theta),-np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]]).dot(xy.T).T



def window_mutual_info(vel,vis,dt0,dt1,vel1=None,select_vis_start=True):
    """
    Mutual information between visible and invisible portions of window. Right now, this only looks
    for information in the sign of the average.
    
    Parameters
    ----------
    vel : ndarray
        Avatar velocity.
    vis : ndarray
    dt0 : float
        Time to look back behind visibility turns on. 
    dt1 : float
        Time to look back after visibility turns on. 
    vel1 : ndarray,None
    select_vis_start : bool,True
    
    Returns
    -------
    mi 
    """
    from entropy import MI,joint_p_mat

    dt0,dt1 = int(dt0*60),int(dt1*60)
    if select_vis_start:
        visStartIx = np.where(np.diff(vis)==1)[0]
    else:
        visStartIx = np.where(np.diff(vis)==-1)[0]
    dt0ix = visStartIx-dt0
    dt1ix = visStartIx+dt1
    
    assert ( (dt0ix<visStartIx) & (visStartIx<dt1ix) ).all()
    
    # Keep indices within bounds of array.
    keepix = (dt0ix>=0) & (dt1ix<len(vel))
    dt0ix,dt1ix = dt0ix[keepix],dt1ix[keepix]
    visStartIx = visStartIx[keepix]
    
    v = np.zeros((len(dt0ix),2))
    if vel1 is None:
        vel1 = vel
    for i,(t0,t1,t2) in enumerate(zip(dt0ix,visStartIx,dt1ix)):
        v[i] = np.sign(vel[t0:t1].mean()),np.sign(vel1[t1:t2].mean())
    
    return MI( joint_p_mat(v,[0],[1]) )

def mean_window_v(avv,vis,dt0,dt1):
    """
    Return velocity trajectory averaged across windows for a single trial.
    
    Parameters
    ----------
    avv : ndarray
        Avatar velocity.
    vis : ndarray
    dt0 : float
        Time to look back behind visibility turns on. 
    dt1 : float
        Time to look back after visibility turns on. 
    
    Returns
    -------
    vMean : ndarray
    vStd : ndarray
    """
    dt0,dt1 = int(dt0*60),int(dt1*60)
    visStartIx = np.where(np.diff(vis)==1)[0]
    dt0ix = visStartIx-dt0
    dt1ix = visStartIx+dt1
    
    # Keep indices within bounds of array.
    keepix = (dt0ix>=0) & (dt1ix<len(avv))
    dt0ix,dt1ix = dt0ix[keepix],dt1ix[keepix]

    v = np.zeros((len(dt0ix),(dt1ix-dt0ix).min()))
    for i,(t0,t1) in enumerate(zip(dt0ix,dt1ix)):
        v[i] = avv[t0:t0+v.shape[1]]
    
    return v.mean(0),v.std(0)

def corrcoef_before_vis(subv,avv,vis,dt):
    """
    Coherence using the wavelet transform for dt seconds around the visibility turning back on.
    
    Parameters
    ----------
    subv : ndarray
        Subject velocity.
    avv : ndarray
        Avatar velocity.
    vis : ndarray
    dt : float
        Time to look back behind or after visibility turns on. Positive values correspond to looking
        after the window.

    Returns
    -------
    """
    # Get indices of points near the end of the invisibility window.
    dtprev = int(dt*60)
    visStartIx = np.where(np.diff(vis)==1)[0]+dtprev
    # Apply bounds.
    visStartIx = visStartIx[(visStartIx>=0)&(visStartIx<len(vis))]
    
    return np.corrcoef(subv[visStartIx],avv[visStartIx])[0,1]

def min_phase_time_shift(freq,subjectAngle,templateAngle,
                         bds=1,
                         dtgrid=np.linspace(0,1,100),
                         fweights=None,
                         ax=None,
                         disp=False):
    """
    Find the global time shift that maximizes the phase density within the threshold.
    
    Parameters
    ----------
    freq : ndarray
    subjectAngle : ndarray
        Subject complex angle.
    templateAngle : ndarray
        Template complex angle.
    dtgrid : ndarray,np.linspace(0,1,100)
    fweights : ndarray
        How much to weight each frequency when calculating the average phase shift. This should be
        (n_freq,n_time).
    bds : float or tuple,1
        Bounds for counting phase difference that is within bounds. This is not properly implemented
        yet.
    disp : bool,False
        
    Returns
    -------
    dtshift : float
        Optimal time shift.
    """
    if fweights is None:
        fweights = np.ones_like(subjectAngle)
    zeroDensity = np.zeros_like(dtgrid)
    zeroDensityNull = np.zeros_like(dtgrid)
    
    for i,dt in enumerate(dtgrid):
        subPhaseAfterShift = shift_phase_by_time(freq,subjectAngle,dt)
        dphase = mod_angle( templateAngle-subPhaseAfterShift )
        zeroDensity[i] = ((np.abs(dphase)<bds)*fweights).sum()
        
        temPhaseAfterShift = shift_phase_by_time(freq,templateAngle,dt)
        dphase = mod_angle( templateAngle-temPhaseAfterShift )
        zeroDensityNull[i] = ((np.abs(dphase)<bds)*fweights).sum()
    
    shiftix = np.argmax(zeroDensity)
#     if (zeroDensity[shiftix]>zeroDensityNull[:shiftix]).all():
#         print "Null model does not reach minimum before solution."
    
    if disp:
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(dtgrid,zeroDensity,'o')
        ax.plot(dtgrid,zeroDensityNull,'ro')
        ax.set(xlabel='dt',ylabel='zero density')
        ax.legend(('Subject','Avatar'),fontsize='small',numpoints=1)
        
    return dtgrid[shiftix]

def optimal_time_shift_freq(dphase,dphase_bds=[-pi/10,pi/10],
                            freqs=np.arange(1,31)*.1,
                            weights=None,
                            return_all=False):
    """
    Find the optimal uniform time shift by shifting by each frequency one at a time.
    
    Parameters
    ----------
    dphase : ndarray
        (n_freq,n_time)
    dphase_bds : tuple,(-pi/10,pi/10)
        Integration domain.
    freqs : ndarray,np.arange(1,31)*.1
        Frequency of each row of dphase.
    weights : ndarray,None
        Weight to assign to each frequency when taking mean over frequencies.
    return_all : bool,False
    
    Returns
    -------
    fopt : float
        Optimal frequency for uniform temporal shift.
    """
    from misc.plot import colorcycle
    avgDensity = np.zeros(len(freqs))
    #fig,ax = plt.subplots()
    #cc = colorcycle(len(freqs))

    # Iterate over subtracting temporal delays from different frequencies.
    for i in xrange(len(freqs)):
        shifteddphase = subtract_freq_phase(i,freqs,dphase)
        if weights is None:
            weights_ = np.ones(len(freqs)-1)/(len(freqs)-1)
        else:
            weights_ = np.delete(weights,i)

        #ax.plot( [((s>dphase_bds[0])&(s<dphase_bds[1])).mean()
        #                         for s in shifteddphase] ,c=cc.next(),lw=2)
        avgDensity[i] = np.array([((s>dphase_bds[0])&(s<dphase_bds[1])).mean()
                                 for s in np.delete(shifteddphase,i,axis=0)]).dot(weights_)
    if return_all:
        return freqs[np.argmax(avgDensity)],avgDensity
    return freqs[np.argmax(avgDensity)]

def phase_and_dphase(x,y,wavelet='cgau1',sample_period=1/60,precision=10):
    """
    Calculate complex phase per frequency per time and dphase for two signals
    using a continuous wavelet transform with the complex Gaussian.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    wavelet : str,'cgau1'
        cgau1 or complex Gaussian gives fine temporal precision but poor frequency localization
        whereas the higher derivatives of the complex gaussian like 'cgau8' give low temporal
        precision and high frequency precision.
    sample_period : float,1/60

    Returns
    -------
    freqs : ndarray
    phasex : ndarray
    phasey : ndarray
    dphase : ndarray
    """
    import pywt

    xcwtmat,freqs = pywt.cwt(x,np.arange(1,100),wavelet,sample_period,precision=precision)
    ycwtmat,freqs = pywt.cwt(y,np.arange(1,100),wavelet,sample_period,precision=precision)

    phasex = np.arctan2( xcwtmat.imag,xcwtmat.real )
    phasey = np.arctan2( ycwtmat.imag,ycwtmat.real )

    dphase = mod_angle(phasex-phasey)

    return freqs,phasex,phasey,dphase

def select_freqs(freqs,precision=1):
    """
    Only return a subset of the frequencies (unique by number of decimal points) and the 
    indices of the selected frequencies.

    Params
    ------
    freqs : ndarray
    precision : int,1

    Returns
    -------
    ufreqs : ndarray
    freqix : ndarray
    """
    freqs = np.around(freqs,precision)
    ufreqs,freqix = np.unique(freqs,return_index=True)
    return ufreqs,freqix  # remember frequencies are return in reverse order

def phase_entropy(d,n_bins=20):
    """
    Calculate entropy for each row in d.

    Params:
    -------
    d (ndarray)
        Each row corresponds to a single frequency.
    n_bins (int=20)

    Returns:
    --------
    S (ndarray)
    """
    S = np.zeros((len(d)))
    for i,row in enumerate(d):
        bins = np.linspace(-np.pi,np.pi,n_bins)

        n = np.histogram(row,bins=bins)[0]
        n = n/n.sum()
        S[i] = np.nansum( -n*np.log2(n) )
    return S

def phase_peakiness(d,bds=.8):
    """
    Calculate density within bds.
    
    Params:
    -------
    d (ndarray)
        Each row corresponds to a single frequency.
    bds (float=0.8)
    
    Returns:
    --------
    peakiness (ndarray)
        Peakiness per given row of d.
    """
    if type(bds) is int or type(bds) is float:
        bds = [-bds,bds]
    
    peakiness = np.zeros((len(d)))
    for i,row in enumerate(d):
        ix = (bds[0]<=row)&(row<=bds[1])
        peakiness[i] = ix.sum()/len(row)
    
    return peakiness

def subtract_freq_phase(subtractIx,f,dphase):
    """
    Subtract lower frequency phase from higher frequencies in temporal domain.

    Params:
    -------
    subtractIx (int)
    f (ndarray)
    dphase (ndarray)
    """
    dphase = dphase.copy()
    dphase /= f[:,None]*2*np.pi
    dphase = dphase-dphase[subtractIx]
    dphase *= f[:,None]*2*np.pi
    dphase = mod_angle(dphase)
    return dphase

def shift_phase_by_time(f,phase,time_shift):
    """
    Time shift is subtracted from given phase. A positive time shift pushes things forwards in time
    and a negative shift pushes things backwards (or later).

    Params:
    -------
    f : ndarray
        Frequencies of each row of phase.
    phase : ndarray
        In units of radians.
    time_shift : float

    Returns
    -------
    phase : ndarray
        Same shape as given phase.
    """
    phase = phase.copy()
    phase -= f[:,None]*2*pi*time_shift
    phase = mod_angle(phase)
    return phase

def find_closest_dtheta(t1,t2,window=101,dt=1):
    """
    Return relative index of closest data point. How much t2 has to be shifted to match t1.

    Params:
    -------
    t1 (ndarray)
    t2 (ndarray)
    window (int=101)
        Width of window for searching for min distance. Must be odd.
    dt (float=1)
        dt spacing corresponding to index.
    """
    assert (window%2)==1
    
    mnix = np.zeros((len(t1)-window),dtype=int)
    for i,tix in enumerate(xrange(window//2,len(t1)-window//2-1)):
        mnix[i] = np.argmin(abs(t2[tix]-t1[tix-window//2:tix+window//2]))-window//2
    return mnix*dt

def project_gaze_to_plane(pitch,yaw,pos,
                          dpitch=None,dyaw=None,dpos=None,
                          distance_to_plane=.5,return_vel=False):
    """
    Given pitch and yaw, project the gaze of the person to a plane in front of him.
    
    Params:
    -------
    p (ndarray)
        Pitch
    y (ndarray)
        Yaw
    pos (ndarray)
        xyz coordinates of headset
    distances_to_plane (float)
        Distance in meters from headset starting position.
    """
    xp = pos[:,0] + distance_to_plane*np.sin(yaw)
    yp = pos[:,1] + distance_to_plane*np.sin(pitch)
    if not return_vel:
        return xp,yp

    xpv = dpos[:,0] + distance_to_plane*np.cos(yaw)*dyaw
    ypv = dpos[:,1] + distance_to_plane*np.cos(pitch)*dpitch
    return xp,yp,xpv,ypv

def match_bool_indices(*args):
    """
    With multiple boolean indices, reduce the number of True values in one (starting from the end) til it
    matches the number of True values in the other. This might be necessary when selecting times from two
    different time series to plot against one another.

    Params:
    -------
    *args (iterable of ndarrays)

    Returns:
    --------
    None
    """
    mn = min([a.sum() for a in args])
    d = [a.sum()-mn for a in args]
    for i,d_ in enumerate(d):
        if d_>0:
            lastix = np.where(args[i])[0][-d_:]
            args[i][lastix] = False

def select_time_index(tbds,t1,t2):
    """
    Given a time range to select, return a boolean index such taht the number of selected entries are the same
    from t1 and from t2. This assumes that there is a close match between the time samples given in t1 and t2.

    Params:
    -------
    tbds (list)
    t1 (ndarray)
    t2 (ndarray)

    Returns:
    --------
    ix1 (ndarray)
    ix2
    """
    ix1 = (tbds[0]<t1) & (t1<tbds[1])
    ix2 = (tbds[0]<t2) & (t2<tbds[1])
    
    ix1sum = ix1.sum()
    ix2sum = ix2.sum()
    while ix1sum!=ix2sum:
        # Should we take from the beginning or the end?
        if abs(t1[ix1][0]-t2[ix2][0]) < abs(t1[ix1][-1]-t2[ix2][-1]):
            if ix1sum>ix2sum:
                ix1[np.argmax(ix1)] = False
                ix1sum -= 1
            else:
                ix2[np.argmax(ix2)] = False
                ix2sum -= 1
        else:  # take from end
            if ix1sum>ix2sum:
                ix1[len(ix1)-1-np.argmax(ix1[::-1])] = False
                ix1sum -= 1
            else:
                ix2[len(ix2)-1-np.argmax(ix2[::-1])] = False
                ix2sum -= 1

    return ix1,ix2

def match_time(x,t,dt,spline_kwargs={},offset=0,use_univariate=False,
               knot_spacing=1/30):
    """
    Given a data set with non-uniform time stamps (as datetime), interpolate it to have the
    given time spacing. Spline values outside of given range are automatically set to 0.
    
    Note: It is not clear how to choose the spline parameters in some optimal way. I am keeping fits
    "proportional" the length of the data by fixing the number of knots to be proportional to the length of
    the data set.

    Parameters
    ----------
    x : ndarray
        n_time x n_dim
    t : ndarray of datetime.datetime
    spline_kwargs : dict
    offset : float
        Offset to time vector.
    use_univariate : bool,False
        By default, use LSQUnivariateSpline which allows us to choose the knots which is useful for comparable
        fits on data sets of different sizes.
    knot_spacing : float,1/30
        Distance between uniformly spaced knots.

    Returns
    -------
    xSpline (UnivariateSpline or MultiUnivariateSpline)
    tLin (ndarray)
    """
    t = t-t[0]
    t = np.array([t_.total_seconds() for t_ in t])+offset
    tLin = np.arange((t[-1]-offset)/dt)*dt+offset
    
    if x.ndim==1:
        if use_univariate:
            xSpline = UnivariateSpline(t,x,ext=1,s=0,**spline_kwargs)
        else:
            xSpline = LSQUnivariateSpline(t,x,np.linspace(t[1],t[-2],int((t[-2]-t[1])/knot_spacing)),
                                          ext=1,check_finite=False,**spline_kwargs)
    else:
        if use_univariate:
            xSpline = MultiUnivariateSpline(t,x,
                                            ext=1,check_finite=False,fit_type='Uni',s=0,
                                            **spline_kwargs)
        else:
            xSpline = MultiUnivariateSpline(t,x,
                                            knot_spacing=knot_spacing,
                                            ext=1,check_finite=False,**spline_kwargs)

    return xSpline,tLin

def phase_d_error(x,y,filt_x_params=None,filt_phase_params=(11,2),noverlap=7/8):
    """
    Relative phase fluctuations per frequency across given signals.
    
    Calculate the phase for two signals for various frequency with a moving window. Take the derivative of 
    the difference of the unwrapped phase to see how much the relative phase fluctuates across the sample.

    Filtering phase after transformation seems to work better rather than the raw phase in simple sin example.

    NOTE: Default filtering parameters are arbitrary.
    2017-03-20
    
    Params:
    -------
    x
    y
    filt_x_params (tuple=None)
        (121,3) is a reasonable choice.
    filt_phase_params (tuple)
    """
    if filt_x_params:
        f,t,spec1,phase1 = spec_and_phase(savgol_filter(x,filt_x_params[0],filt_x_params[1]),noverlap)
        f,t,spec2,phase2 = spec_and_phase(savgol_filter(y,filt_x_params[0],filt_x_params[1]),noverlap)
    else:
        f,t,spec1,phase1 = spec_and_phase(x,noverlap)
        f,t,spec2,phase2 = spec_and_phase(y,noverlap)
    
    # Filtering phase seems to be important to get a good estimate. Have only tested this with a simple sine
    # signal.
    # Ignore 0 frequency.
    f = f[1:]
    #phase1 = np.unwrap(phase1[1:],axis=1)
    #phase2 = np.unwrap(phase2[1:],axis=1)
    phase1 = savgol_filter(np.unwrap(phase1[1:],axis=1),
                           filt_phase_params[0],
                           filt_phase_params[1],axis=1)
    phase2 = savgol_filter(np.unwrap(phase2[1:],axis=1),
                           filt_phase_params[0],
                           filt_phase_params[1],axis=1)
    
    # Cumulative error in the derivative.
    cumerror = np.abs(np.diff(phase1-phase2)).sum(1)
    return f,t,cumerror/len(t),phase1,phase2

def spec_and_phase(X,noverlap,
                   dt=1/120,
                   window=('gaussian',50),
                   nperseg=501,
                   pos_freq=True):
    """
    Compute spectrogram and the corresponding phase for a 1D signal. This can be used to look at phase
    coherence.

    Params:
    -------
    X (ndarray)
        1d signal
    noverlap (float)
        Fraction of overlap.
    dt (float=1/120)
    window (tuple,('gaussian',50))
        For scipy.signal.get_window()
    nperseg (int=501)
    pos_freq (bool=True)
        If true, return only positive frequencies.

    Value:
    ------
    f (ndarray)
        Frequencies.
    t (ndarray)
        Times at which spectra occur.
    spec (ndarray)
        Spectrogram. (n_freq,n_time)
    phase (ndarray)
        Array of phases for each frequency.
    """
    from filter import spectrogram
    from scipy.signal import get_window

    assert noverlap<1
    noverlap = int(noverlap*nperseg)

    window = get_window(window,nperseg)
    f,t,spec = spectrogram(X,window,nperseg-noverlap,fs=1/dt,npadding=nperseg//2)
    if pos_freq:
        spec = spec[f>0]
        f = f[f>0]

    phase = np.angle(spec)
    return f,t,spec,phase

def quaternion_to_rot(q,normalize=False):
    """
    Convert quaternion to a four-vector where first entry is the rotation about the unit vector given by
    the last three elements of the quaternion.

    Params:
    -------
    q (ndarray)
        n_samples x 4 list of quaternions
    normalize (bool=False)
        Normalize the rotation vector for the quaternion to erase numerical precision errors.
    """
    phi = np.arccos(q[:,0])*2
    u = q[:,1:]/sin(phi[:,None]/2)
    if normalize:
        u /= np.linalg.norm(u,axis=1)[:,None]  # Normalize this to one. It should normalized already, so just
                                               # getting rid of numerical precision errors.
    return np.hstack((phi[:,None],u))

def optimize_time(t1,x1,t2,x2,
                  scale=1.0,
                  scale_bounds=[.98,1.02],
                  fix_scale=False,
                  offset=-5,
                  max_offset=10,
                  method='minimize',
                  full_output=False):
    """
    Find best way to overlap curves by rescaling and translating in time. This is used primarily for Vicon and
    Perception Neuron comoparisons.
    Cost function for finding optimal time scale factor and offset between two data sets.
    Offset and scaling are applied to second set of given trajectories.
    
    Params:
    -------
    t1 (ndarray)
        Time should be ordered.
    x1 (list of ndarrays)
    t2 (ndarray)
    x2 (list of ndarrays)
    scale (float=1.0)
    scale_bounds (list)
    fix_scale (bool=False)
        If True, scale will be fixed at given value.
    offset (float=-5)
    max_offset (float)
    method (str='minimize')
        'grid','minimize'
    full_output (bool=False)

    Returns:
    --------
    scale (float)
    offset (float)
    """
    if type(offset) is float or type(offset) is int:
        assert abs(offset)<max_offset, "Given offset is outside of bounds."
    else:
        assert abs(offset[-1])<max_offset, "Given offset is outside of bounds."

    def f(offset=offset,scale=scale):
        if scale>scale_bounds[1] or scale<scale_bounds[0]:
            return [1e30]
        if abs(offset)>max_offset:
            return [1e30]
        t2Transform = t2*scale + offset
        
        cost = 0.
        for x1_,x2_ in zip(x1,x2):
            # Assuming that the second set of data is small enough to be shifted and scaled within the time
            # range fo the first data set. And that the times are all ordered.
            t1Ix = (t1>t2Transform[0]) & (t1<t2Transform[-1])

            interp = interp1d(t2Transform,x2_,axis=0,bounds_error=True)
            thisCost = np.linalg.norm( interp(t1[t1Ix]) - x1_[t1Ix] )
            cost += thisCost
        return cost
    
    # Simple grid search.
    if method=='grid':
        if fix_scale:
            cost = np.zeros_like(offset)
            for i,o in enumerate(offset):
                cost[i] = f(o)
            minix = np.argmin(cost)
            output = [offset[minix]]
        else:
            cost = np.zeros_like(offset)
            for i in xrange(scale.shape[0]):
                for j in xrange(offset.shape[1]):
                    scale,offset = scale[i,j],offset[i,j]
                    cost[i,j] = f(offset,scale)
            minix = np.argmin(cost)
            output = [scale[minix[0]],offset[minix[1]]]

        if full_output:
            output.append(cost)
        return tuple(output)
    else:
        if fix_scale:
            soln = minimize(lambda params: f(*params),offset,method=method )
            offset = soln['x']
            return offset
        else:
            soln = minimize(lambda params: f(*params),[offset,scale],method=method )
            offset,scale = soln['x']
            return scale,offset

def phase_lag(v1,v2,maxshift,windowlength,dt=1,measure='dot',window=None,v_threshold=0):
    """
    Find index shift that would maximize the overlap between two different time series. This involves taking
    windows of one series and moving across the other time series to find maximal agreement.
    2017-03-27

    Params:
    -------
    v1,v2 (ndarray)
        n_samples x n_dim. Time series to compare. v2 will be windowed and shifted around meaning that
        positive phase lags are when it's behind v1 and negative phase lags are when it's ahead.
    maxshift (int)
        Maximum phase shifting allowed.
    windowlength (int)
        Width of window to use.
    measure (str='dot')
        Use 'dot' product or 'corr' correlation coefficient.
    dt (float=1)
        Amount of time that each index increment corresponds to. This determines the units that the phase is
        returned as.
    v_threshold (float=0)
        Minimum norm below which to set vector value to 0.

    Returns:
    --------
    phase
        Phase difference in units of dt.
    overlaperror
        Max overlap measure used to determine phase lag. This is in the interval [-1,1].
    """
    if window is None:
        filtwindow = np.ones((windowlength,1))
    else:
        filtwindow = window[:,None] 
    v1,v2 = v1.copy(),v2.copy()

    if measure=='dot':
        assert v1.ndim>1 and v1.shape[1]>1, "Dot option was written for vectors only."
        normv1,normv2 = norm1(v1),norm1(v2)
        nanix1,nanix2 = normv1<v_threshold,normv2<v_threshold
        v1[nanix1] = np.nan
        v2[nanix2] = np.nan

        v1=v1/normv1[:,None]
        v2=v2/normv2[:,None]
        
        # Label the zero vectors to always give max error in the dot product.
        v1[np.isnan(v1)] = -1
        v2[np.isnan(v2)] = -1

        def f(i):
            """
            Window is the selection in the first vector that we hold fixed. Then we take a moving window
            across the second vector and see how good the overlap is between the two vectors.
            """
            window=v2[i:i+windowlength]
            overlapcost=np.zeros((2*maxshift))  # average overlap between the two velocity time series

            # Shift background.
            for j in xrange(maxshift*2):
                background = v1[i-maxshift+j:i-maxshift+windowlength+j]
                dotprod = (window*background).sum(1)
                dotprod[dotprod>1] = 1  # thresholded 0 vectors should be perfectly aligned as defined by 1
                overlapcost[j] = (dotprod*filtwindow).mean()

            # Look for local max starting from the center of the window where delay=0.
            maxix = local_argmax(overlapcost,windowlength//2)
            phase = (maxix-maxshift)*-dt
            overlaperror = overlapcost[maxix]
            return phase,overlaperror

        p = mp.Pool(mp.cpu_count())
        phase,overlaperror = zip(*p.map(f,range(maxshift,len(v1)-maxshift-windowlength)))
        phase,overlaperror = np.array(phase),np.array(overlaperror)
        p.close()

    elif measure=='corr':
        # Normalized correlation E[x*y]/sqrt(E[x^2]E[y^2]). This accounts for importance of the sign by not
        # subtracting off the mean.
        
        # Define function for calculating phase lag for each dimension separately.
        def _calc_single_col(v1,v2):
            phase = np.zeros((len(v1)-2*maxshift-windowlength))
            overlaperror = np.zeros((len(v1)-2*maxshift-windowlength))
            L = windowlength+maxshift*2
            
            counter = 0
            for i in xrange(maxshift,len(v1)-maxshift-windowlength):
                window = v2[i:i+windowlength]
                background = v1[i-maxshift:i+maxshift+windowlength]
                overlapcost = crosscorr(background,window)
                if overlapcost.ndim>1:
                    overlapcost = overlapcost.sum(1)
                
                # Look for local max starting from the center of the window where delay=0.
                maxix = local_argmax(overlapcost,L//2)
                phase[counter] = (maxix-L//2)*-dt
                overlaperror[counter] = overlapcost.max()
                counter += 1 
            return phase,overlaperror
        
        # Calculate phase lag by looping through all dimensions.
        phase,overlaperror = _calc_single_col(v1,v2)

    else: raise Exception("Bad correlation measure option.")

    return phase,overlaperror

def crosscorr(background,window,subtract_mean=False):
    """Normalized cross corelation from moving window across background. Remember that when this window is
    oved across, we must reverse the order in which the array is read."""
    ones = np.ones_like(window)/len(window)
    window = window[::-1]
    
    if subtract_mean:
        windowMean = window.mean(0)
        backgroundMean = fftconvolve_md( background,args=[ones] )
        backgroundSquare = fftconvolve_md( background**2,args=[ones] )
        
        num = fftconvolve_md(background,args=[window])/len(window) - backgroundMean*windowMean
        denom = np.sqrt(backgroundSquare-backgroundMean**2)*window.std(0)

        overlapcost = num/denom
    else:
        windowabsmean = np.sqrt( (window*window).mean(0) )
        backgroundabsmean = np.sqrt( fftconvolve_md(background**2,
                                                    args=[ones]) )
        overlapcost = fftconvolve_md(background,args=[window])/len(window) / (windowabsmean * backgroundabsmean)
    return overlapcost

def fftconvolve_md(x,args=[],axis=0):
    """
    fftconvolve on multidimensional array along particular axis
    """
    if x.ndim>1:
        if axis==0:
            conv = np.zeros_like(x)
            if args[0].ndim>1:
                for i in xrange(x.shape[1]):
                    conv[:,i] = fftconvolve(x[:,i],args[0][:,i],mode='same')
                return conv
            else:
                for i in xrange(x.shape[1]):
                    conv[:,i] = fftconvolve(x[:,i],args[0],mode='same')
                return conv
        else:
            raise NotImplementedError
    return fftconvolve(x,args[0],mode='same')

@jit
def norm1(x):
    """Helper function for phase_lag()"""
    return np.sqrt((x*x).sum(axis=1))

@jit(nopython=True)
def local_argmax(x,ix0):
    """
    Find biggest local max starting from ix0 and looking in both directions.
    2017-02-18
    """
    # Left side first.
    atMax= False
    ix = ix0
    while not atMax:
        if x[ix-1]>x[ix]:
            ix -= 1
        else:
            atMax = True
        if ix==0:
            atMax = True
    leftMaxIx = ix
    
    # Search right side.
    atMax = False
    ix = ix0
    while not atMax:
        if x[ix+1]>x[ix]:
            ix += 1
        else:
            atMax = True
        if  ix==(len(x)-1):
            atMax = True
    rightMaxIx = ix
    
    if x[leftMaxIx]>=x[rightMaxIx]:
        return leftMaxIx
    else:
        return rightMaxIx

def get_reshape_samples(sample,neighborsix,windowlen):
    """
    Return samples from list of samples but reshaped as sample_length x n_dim.
    2016-12-09
    """
    samples_ = sample[neighborsix]
    samples_ = samples_.reshape(len(samples_),windowlen,3)
    return samples_

def get_samples(X,samplesix,windowlen,minusdt,plusdt):
    """
    Fetch specified samples of window length starting at samplesix and padded
    from data.
    2016-12-09

    Params:
    -------
    X (ndarray)
        n_samples x n_dim
    samplesix (list of ints)
        Locations of samples to extract.
    windowlen (int)
    minusdt (int)
        Left side padding
    plusdt (int)

    Value:
    ------
    sample (ndarray(
        n_samples x sample_length x 3
    """
    sample = np.zeros((len(samplesix),windowlen+minusdt+plusdt+1,3))
    for i,ix in enumerate(samplesix):
        sample[i] = X[ix-minusdt:ix+windowlen+plusdt+1]
    return sample

def find_neighbors(samples,distThresh,neighborsSep=30):
    """
    Find set of nearest neighbors below a threshold (Euclidean metric) for each sample. Nearest neighbors
    should be separated by at least n frame.
    2016-12-09
    
    Params:
    -------
    distThresh (float)
        Cutoff for determining whether or not two samples are neighbors.
    neighborSep (int=30)
        Min separation between two found neighbors. This prevents the algorithm from returning basically the
        same sample multiple times as a neighbor.

    Value:
    ------
    neighbors
        List of neighbors for each given sample including self as a neighbor.
    nNneighbors
        Length of each list of neighbors.
    """
    from numpy.linalg import norm

    neighbors = []
    for i in xrange(len(samples)):
        ix = np.argwhere(norm(samples[i][None,:]-samples,axis=1)<distThresh).ravel()
        if len(ix)>0:
            ix = np.concatenate([[ix[0]],ix[np.argwhere(np.diff(ix)>neighborsSep).ravel()+1]])
        neighbors.append(ix)
    nNeighbors = np.array([len(i) for i in neighbors])
    return neighbors,nNeighbors

def sliding_sample(X,windowlen,):
    """
    Take samples from the given data sliding a flat window across it and taking all data points that fall into
    the window.
    2016-12-09
    
    Values:
    -------
    X (ndarray)
        n_samples x n_dim
    windowlen (int)
        Width of sampling window.
    """
    samples = np.zeros((len(X)-windowlen,X.shape[1]*windowlen))
    for i in xrange(len(X)-windowlen):
        samples[i] = X[i:i+windowlen].ravel()
    return samples

def initial_orientation(df):
    """
    Using the point between the hands and the z-vector to get the vector pointing between the two subjects.
    Center the dataframe to the midpoint between the two hands in the xy plane.

    z-axis points towards the ground.
    2016-12-10

    Params:
    -------
    df (pandas.DataFrame)
        With XVA columns.
    """
    skeleton = load.calc_file_body_parts() 
    upperix = skeleton.index('left foot contact')
    
    # Vector from the position of the left to the right hand at the initial stationary part.
    handsIx = [skeleton.index('LeftHand'),skeleton.index('RightHand')]
    handsvector = ( df.iloc[:10,handsIx[1]*9:handsIx[1]*9+3].mean(0).values-
                    df.iloc[:10,handsIx[0]*9:handsIx[0]*9+3].mean(0).values )
    handsvector[-1] = 0.

    # Position between the hands.
    midpoint = ( df.iloc[:10,handsIx[1]*9:handsIx[1]*9+3].mean(0).values+
                 df.iloc[:10,handsIx[0]*9:handsIx[0]*9+3].mean(0).values )/2
    midpoint[-1] = 0
    for i in xrange(upperix):
        # Only need to subtract midpoint from X values which are the first three cols of each set.
        df.values[:,i*9:i*9+3] -= midpoint[None,:]

    bodyvec = np.cross(handsvector,[0,0,-1.])
    bodyvec /= np.linalg.norm(bodyvec)
    return bodyvec

def moving_mean_smooth(x,filtDuration=12):
    """
    Wrapper for moving mean.
    2016-11-09
    
    Params:
    -------
    x (vector)
    filtDuration (float)
        Moving mean filter duration in number of consecutive data points.
    """
    if x.ndim>1:
        y = np.zeros_like(x)
        for i in xrange(x.shape[1]):
            y[:,i] = fftconvolve( x[:,i],np.ones((filtDuration)),mode='same' )
        return y
    return fftconvolve(x,np.ones((filtDuration)),mode='same')

def discrete_vel(v,timescale):
    """
    Smooth velocity, sample with frequency inversely proportional to width of moving
    average and take the sign of the diff.
    2016-11-09

    Params:
    -------
    v (vector)
    timescale (int)
    """
    # I commented out time average smoothing because it confuses the analysis for what information transfer
    # actually means. It might be better to use Savitzky-Golay filtering or something like that. However,
    # those are also fit by looking over many data points.
    #vsmooth = moving_mean_smooth(v,timescale)[::timescale]
    vsmooth = v[::timescale]

    if v.ndim>1:
        change = np.sign(np.diff(vsmooth,axis=0))
    else:
        change = np.sign(np.diff(vsmooth))
    
    return change

def convert_t_to_bins(timescale,dt):
    """
    Convert timescale given in units of seconds to bins for a moving average in the given sample. Remove
    duplicate timescale entries that appear when this discretization to bins has occurred. Only return lags
    that are greater than 0.
    2016-11-07

    Params:
    -------
    timescale (int)
    dt (float)
    """
    discretetimescale = (timescale/dt).astype(int)
    discretetimescale = discretetimescale[discretetimescale>0]
    ix = np.unique(discretetimescale,return_index=True)[1]
    return discretetimescale[ix],discretetimescale[ix]*dt

def asym_mi(leader,follower,timescale):
    """
    Asymmetric MI using moving average.
    2016-11-07
    
    Params:
    -------
    leader (vector)
    follower (vector)
    timescale (vector)
        Given in units of the number of entries to skip in sample.
    """
    lchange = discrete_vel(leader,timescale)
    fchange = discrete_vel(follower,timescale)

    p = info.get_state_probs(np.vstack((lchange,fchange)).T,
                             allstates=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])).reshape((2,2))
    samemi = info.MI(p)

    p = info.get_state_probs(np.vstack((lchange[:-1],fchange[1:])).T,
                             allstates=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])).reshape((2,2))
    ltofmi = info.MI(p)

    p = info.get_state_probs(np.vstack((lchange[1:],fchange[:-1])).T,
                             allstates=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])).reshape((2,2))
    ftolmi = info.MI(p)

    return samemi,ltofmi,ftolmi

def transfer_info(leader,follower,timescale):
    """
    Wrapper for computing transfer entropy between two trajectories using a binary representation.
    2016-11-09
    
    Params:
    -------
    leader (vector)
    follower (vector)
    timescale (vector)
        Given in units of the number of entries to skip in sample.
    """
    from entropy.transfer import TransferEntropy
    te = TransferEntropy()
    
    lchange = discrete_vel(leader,timescale)
    fchange = discrete_vel(follower,timescale)
    
    if lchange.ndim>1:
        lchange = unique_rows(lchange,return_inverse=True)
    else:
        lchange = unique_rows(lchange[:,None],return_inverse=True)
    if fchange.ndim>1:
        fchange = unique_rows(fchange,return_inverse=True)
    else:
        fchange = unique_rows(fchange[:,None],return_inverse=True)

    ltofinfo = te.n_step_transfer_entropy(lchange,fchange,discretize=False)
    ftolinfo = te.n_step_transfer_entropy(fchange,lchange,discretize=False) 
    return ltofinfo,ftolinfo

def truncate(t,y,t0=10,t1=10):
    """
    Truncate ends of data set assuming that y is (n_time,n_dim).

    Params:
    -------
    t (ndarray o pd.DataFrame)
    y (ndarray)
    t0 (float=10)
    t1 (float=10)

    Returns:
    --------
    yTruncated
    """
    timeix = np.logical_and(t>=t0,t<=(t[-1]-t1))
    if type(y) is pd.core.frame.DataFrame:
        if y.ndim==2:
            return y.iloc[timeix,:]
        else:
            return y.iloc[timeix]
    else:
        if y.ndim==2:
            return y[timeix,:]
        else:
            return y[timeix]

def spline_smooth(t,Y,T=None,S=None,fixedKnots=True,spline_kwargs={}):
    """
    Use quintic least squares spline to smooth given data in place down the columns. Knots appear about every
    second as estimated from the inverse sampling rate.
    2016-11-09

    Params:
    -------
    t (vector)
        Time of measurements
    Y (ndarray) 
        n_time x n_dim. Measurements.
    T (int=None)
        Indices between knots.
    S (float=None)
        Smoothing factor for UnivariateSpline.
    fixedKnots (bool=True)
        If true, use the sampling rate as the default spacing between knots or the given T. Otherwise, use
        smoothing factor method UnivariateSpline.
        
    Value:
    ------
    spline (list)
        List of LSQUnivariateSpline instances for each col of input Y.
    """
    if fixedKnots:
        if T is None:
            dt = t[1]-t[0]
            T = int(1/dt)
        spline = []
        for i,y in enumerate(Y.T):
            spline.append( LSQUnivariateSpline(t,y,t[T::T],k=5) )
            Y[:,i] = spline[-1](t)
    else:
        spline = []
        for i,y in enumerate(Y.T):
            spline.append( UnivariateSpline(t,y,k=5,s=S,**spline_kwargs) )
            Y[:,i] = spline[-1](t)

    return spline

def Ryxz(a,b,c):
    """
    BVH rotation matrix. As from bvhplayer's skeleton.py process_bvhkeyframe(). For multiplication on the
    right side of the vector to be transformed. These rotation matrices are the canonical rotation matrices
    transposed (for left side multiplication).
    The convention as stated in http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf does not
    correspond to that used in Perception Neuron. I tested this by collecting some data with my wrist moving
    in a circle. Reversing the order of the multiplication gives weird results whereas this one gives the
    correct results.
    2016-10-24
    """
    ry = np.array([[cos(a),0,-sin(a)],[0,1,0],[sin(a),0,cos(a)]]).T
    rx = np.array([[1,0,0],[0,cos(b),sin(b)],[0,-sin(b),cos(b)]]).T
    rz = np.array([[cos(c),sin(c),0],[-sin(c),cos(c),0],[0,0,1]]).T
    return ry.dot(rx.dot(rz))

def rotate_by_angles(v0,a,b,c,*args):
    """
    Rotate v0 by given set of angles. Assuming given Euler angles are in order of YXZ for left hand multiplication.
    2016-10-23
    
    Params:
    -------
    v0
    a,b,c (vectors)
        Rotation angles (y,x,z).
    *args (vectors)
        More rotation angles. Children after the parents. Because this means that the children are multiplied
        further to the left and precede the vector to be rotated immediately.
    """
    assert (len(args)%3)==0

    v = np.zeros((len(a),3))
    if len(args)>0:
        for i in xrange(len(a)):
            R = Ryxz(a[i],b[i],c[i])
            for j in xrange(len(args)//3):
                R = R.dot(Ryxz(args[j*3][i],args[j*3+1][i],args[j*3+2][i]))
            v[i] = v0.dot(R)
    else:
        for i in xrange(len(a)):
            v[i] = v0.dot(Ryxz(a[i],b[i],c[i]))

    return v

def polar_angles(v0,a,b,c):
    """
    Get the polar and azimuthal angles that correspond to the new position of v0 after applying rotation 
    matrices with angles a,b,c.
    With YXZ.v0 convention.
    2016-10-23
    
    Params:
    -------
    v0 (floats)
        Starting vector.
    a,b,c (floats)
        Euler angles about x,y,z axes.
    """
    v1 = v0.dot(Ryxz(a,b,c))
    phi = np.arctan2( v1[1],v1[0] )
    theta = np.arccos( v1[2] )
    return v1,phi,theta

def convert_euler_to_polar(yxz):
    """
    Convert Euler angles into polar/azimuthal angles.
    2016-10-23
    
    Params:
    -------
    yxz (ndarray)
        With columns for rotation angles about y,x,z axes.

    Value:
    ------
    phis,thetas
    """
    if type(yxz) is pd.core.frame.DataFrame:
        yxz = yxz.values
    elif not type(yxz) is np.ndarray:
        raise Exception("Unexpected data type.")

    v0 = np.array([0,0,1.])
    phis,thetas = np.zeros((len(yxz))),np.zeros((len(yxz)))
    for i,r in enumerate(yxz):
        v,phis[i],thetas[i] = polar_angles(v0,r[1],r[0],r[2])

    # Account for discontinuities.
    phis = np.unwrap(phis,discont=2*np.pi)
    thetas = np.unwrap(thetas)

    return phis,thetas

def euler_to_vectors(*angles):
    from ising.heisenberg import rotate

    x0=np.array([0,0,1.])
    x = rotate_by_angles(x0,*angles)
    
    # Normalize vector and rotate so that we can project easily onto x-y plane as defined
    # by average vector.
    xavg = x.mean(0)
    xavg /= np.linalg.norm(xavg)
    x /= np.linalg.norm(x,axis=1)[:,None]
    
    n = np.cross(xavg,np.array([0,0,1.]))
    d = np.arccos(xavg[-1])
    xavg = rotate( xavg, n, d )
    x = rotate(x,n,d)
    return x

def extract_phase(*angles):
    """
    Extract phase given the Euler angles as output by BVH. Phase is defined as the xy angle in the 
    plane of the average vector.
    2016-10-23
    """
    x = euler_to_vectors(*angles)
    
    return np.unwrap(np.arctan2(x[:,1],x[:,0]),discont=np.pi)

def train_cal_noise(leaderW,followerW,dv,nTrainSamples=None):
    """
    Train Gaussian process noise prediction on hand calibration trials (with hands touching) using the angular
    velocities.
    2017-01-31

    Params:
    -------
    leaderW, followerW (ndarray)
        (n_samples, n_dim)
    dv (ndarray)
        (n_samples, 3) Error in velocities.
    trainSamples (int=None)
        Number of training samples to use.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor

    X = np.hstack((leaderW,followerW))
    Y = dv
    nTrainSamples = nTrainSamples or len(X)//2
    assert 0<nTrainSamples<=len(X)
   
    randix = np.zeros((len(X)))==1
    randix[np.random.choice(range(len(X)),size=nTrainSamples)] = True
    trainX = X[randix]
    testX = X[randix==0]
    trainY = Y[randix]
    testY = Y[randix==0]
    
    gpr = GaussianProcessRegressor(alpha=1e-1,n_restarts_optimizer=10)
    gpr.fit(trainX,trainY)
    
    return gpr,np.corrcoef(gpr.predict(testX).ravel(),testY.ravel())[0,1]



# ===== #
# Tests #
# ===== #
def test_rotate_xy():
    assert np.isclose( rotate_xy(np.array([[0,1],[1,0],[0,-1],[-1,0]]),np.pi/2),
                       np.array([[-1,0],[0,1],[1,0],[0,-1]]) ).all()

