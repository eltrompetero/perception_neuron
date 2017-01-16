# Edward Lee edl56@cornell.edu
# 2016-08-11
from __future__ import division
try:
    import matplotlib.pyplot as plt
except ImportError:
    print "Could not import matplotlib."
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin,cos
import pandas as pd
from ising.heisenberg import rotate
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline
import entropy.entropy as info
from scipy.signal import fftconvolve
from misc.utils import unique_rows
import load

# ---------------------- #
# Calculation functions. #
# ---------------------- #
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
    Fetch specified samples of window length starting at samplesix and padded from data.
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

    It is not clear whether the z-axis is in the global up or pointing towards the ground.
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

    bodyvec = np.cross(handsvector,[0,0,1.])
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
    Truncate ends of data set.
    2016-11-07
    """
    timeix = np.logical_and(t>t0,t<(t[-1]-t1))
    if type(y) is pd.core.frame.DataFrame:
        if y.ndim==2:
            return y.ix[timeix,:]
        else:
            return y.ix[timeix]
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
    BVH rotation matrix. As from bvhplayer's skeleton.py process_bvhkeyframe(). For multiplication on the right side of the vector to be transformed. These rotation matrices are the canonical rotation matrices transposed (for left side multiplication).
    The convention as stated in http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf does not correspond to that used in Perception Neuron. I tested this by collecting some data with my wrist moving in a circle. Reversing the order of the multiplication gives weird results whereas this one gives the correct results.
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
        Rotation angles.
    *args (vectors)
        More rotation angles. Children after the parents. Because this means that the children are multiplied further to the left and precede the vector to be rotated immediately.
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
# End calculation files



# ------------------- #
# Plotting functions. #
# ------------------- #
def plot_xva_comparison(fig,ax,x1,x2,v1,v2,a1,a2,aOffset=0.,title=''):
    """
    Plot XVA comparison plots.
    2016-12-14
    """
    ax[0].plot(x1,x2,'.',alpha=.2)
    ax[0].plot([-1,1],[-1,1],'k-')
    ax[0].set(xlabel='Leader pos',ylabel='Follower pos',
              xlim=[-1,1],ylim=[-1,1])
    [l.set_rotation(75) for l in ax[0].xaxis.get_ticklabels()]
    ax[0].text(-.95,.85,"%1.2f"%np.corrcoef(x1,x2)[0,1],fontsize='x-small')
    
    ax[1].plot(v1,v2,'.',alpha=.2)
    ax[1].plot([-1,1],[-1,1],'k-')
    ax[1].set(xlabel='Leader vel',ylabel='Follower vel',
              xlim=[-1,1],ylim=[-1,1])
    [l.set_rotation(75) for l in ax[1].xaxis.get_ticklabels()]
    ax[1].text(-.95,.85,"%1.2f"%np.corrcoef(v1,v2)[0,1],fontsize='x-small')

    ax[2].plot(a1,a2,'.',alpha=.2)
    ax[2].plot([-1.5,1.5],[-1.5,1.5],'k-')
    ax[2].set(xlim=[-.4+aOffset,.4+aOffset],ylim=[-.4+aOffset,.4+aOffset])
    ax[2].set(xlabel='Leader acc',ylabel='Follower acc')
    [l.set_rotation(75) for l in ax[2].xaxis.get_ticklabels()]
    ax[2].text(-.35+aOffset,.35+aOffset,"%1.2f"%np.corrcoef(a1,a2)[0,1],fontsize='x-small')
    
    fig.subplots_adjust(wspace=.5)
    fig.text(.3,.95,title)

def plot_va_comparison(fig,ax,v1,v2,a1,a2,aOffset=0.,title=''):
    ax[0].plot(v1,v2,'.',alpha=.2)
    ax[0].plot([-1,1],[-1,1],'k-')
    ax[0].set(xlabel='Leader vel',ylabel='Follower vel',
              xlim=[-1,1],ylim=[-1,1])
    [l.set_rotation(75) for l in ax[0].xaxis.get_ticklabels()]

    ax[1].plot(a1,a2,'.',alpha=.2)
    ax[1].plot([-1.5,1.5],[-1.5,1.5],'k-')
    ax[1].set(xlim=[-.4+aOffset,.4+aOffset],ylim=[-.4+aOffset,.4+aOffset])
    ax[1].set(xlabel='Leader acc',ylabel='Follower acc')
    [l.set_rotation(75) for l in ax[1].xaxis.get_ticklabels()]
    
    fig.subplots_adjust(wspace=.5)
    fig.text(.3,.95,title)

def plot_hips_drift(hips,dt):
    gs = gridspec.GridSpec(2,3,wspace=.3)
    gs.update(wspace=.4)

    fig = plt.figure(figsize=(12,4))
    ax = [fig.add_subplot(gs[:,:-1]),fig.add_subplot(gs[0,-1]),fig.add_subplot(gs[1,-1])]

    # Draw map as if I were looking down from above. The front faces to the left.
    ax[0].plot(hips['xx']/100,-hips['zz']/100,'.',alpha=.3)
    ax[0].plot(hips['xx'].iloc[0]/100,-hips['zz'].iloc[0]/100,'ro',ms=15)
    ax[0].plot(hips['xx'].iloc[-1]/100,-hips['zz'].iloc[-1]/100,'r+',ms=15,mew=4)
    ax[0].set(xlabel='front-back (m)',ylabel='sideways (m)')
    
    tmax = len(hips['xx'])*dt
    ax[1].plot(np.arange(len(hips['xx']))*dt,hips['xx']/100)
    ax[1].set(xticklabels=[],xlim=[0,tmax],ylabel='front-back')
    [l.set_fontsize(10) for l in ax[1].get_yticklabels()]
    ax[2].plot(np.arange(len(hips['xx']))*dt,-hips['zz']/100)
    ax[2].set(xlabel='time (s)',xlim=[0,tmax],ylabel='sideways')
    [l.set_fontsize(10) for l in ax[2].get_yticklabels()]
    
    print "Drift front-back: %1.3f m"%( (hips['xx'].iloc[-1]-hips['xx'].iloc[0])/100 )
    print "Drift sideways: %1.3f m"%( -(hips['zz'].iloc[-1]-hips['zz'].iloc[0])/100 )
    
    return fig

def plot_positions(bp):
    """
    Plot the 3D positions of a particular body part. The default measurement in Perception Neuron ignores the displacement except for the hips.
    2016-08-11

    Params:
    -------
    bp (pd.DataFrame)
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    
    ax.plot(bp['xx']/100,bp['zz']/100,bp['yy']/100,'o',alpha=.3)
    ax.set(xlim=[bp['xx'].min()/100,bp['xx'].max()/100],
           ylim=[bp['zz'].min()/100,bp['zz'].max()/100],
           zlim=[bp['yy'].min()/100,bp['yy'].max()/100])
    return fig

def plot_euler_angles(t,angles,setkwargs={},linestyles=['-','--','-.']):
    """
    2016-10-28
    """
    fig,ax = plt.subplots(figsize=(12,4))
    for i,a in enumerate(angles):
        ax.plot(t,a[:,0],'b'+linestyles[i])
        ax.plot(t,a[:,1],'g'+linestyles[i])
        ax.plot(t,a[:,2],'r'+linestyles[i])
    ax.set(xlim=[t[0],t[-1]],ylim=[-np.pi,np.pi],xlabel='Time',ylabel='Euler angle',
           yticks=[-np.pi,-np.pi/2,0,np.pi/2,np.pi],yticklabels=[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])
    ax.set(**setkwargs)
    ax.legend(('y','x','z'),fontsize='small',bbox_to_anchor=[1.15,1])
    return fig

def plot_polar_angles(phis,thetas,dt):
    """
    2016-08-12
    
    Params:
    -------
    phis (ndarray)
        Azimuthal angles
    thetas (ndarray)
        Polar angles
    dt (float)
        Timestep between phi and theta data points.
    """
    gs = gridspec.GridSpec(2,2,height_ratios=(1,3),wspace=.2,hspace=.4)
    fig = plt.figure(figsize=(10,4))
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]

    ax[0].plot(dt*np.arange(len(phis)),phis)
    ax[0].set(xlim=[0,len(phis)*dt],ylim=[-np.pi,np.pi],
              xlabel='time (s)',ylabel=r'$\phi$',
              yticks=[-np.pi,0,np.pi],yticklabels=[r'$-\pi$',r'$0$',r'$\pi$'])
    [l.set_fontsize(10) for l in ax[0].get_yticklabels()]
    [l.set_fontsize(10) for l in ax[0].get_xticklabels()]
    ax[0].xaxis.get_label().set_fontsize(10)
    ax[1].plot(dt*np.arange(len(phis)),thetas)
    ax[1].set(xlim=[0,len(phis)*dt],ylim=[0,np.pi],
              xlabel='time (s)',ylabel=r'$\theta$',
              yticks=[0,np.pi/2,np.pi],yticklabels=[r'$0$',r'$\pi/2$',r'$\pi$'])
    [l.set_fontsize(10) for l in ax[1].get_yticklabels()]
    [l.set_fontsize(10) for l in ax[1].get_xticklabels()]
    ax[1].xaxis.get_label().set_fontsize(10)

    ax[1].plot(dt*np.arange(len(phis)),thetas)
    ax[2].plot(phis,thetas,'.',alpha=.2)
    ax[2].set(xlim=[-np.pi,np.pi],ylim=[0,np.pi],xlabel=r'$\phi$',ylabel=r'$\theta$');
    return fig

def plot_unit_trajectory(*angles):
    """
    Extract phase given the Euler angles as output by BVH. Phase is defined as the xy angle in the 
    plane of the average vector.
    2016-10-23
    """
    x = euler_to_vectors(*angles)
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,2],alpha=.8,lw=10)
#     ax.scatter(x[:,0],x[:,1],x[:,2],alpha=.8,lw=0,s=1000,
#                c=[plt.cm.copper(i/len(x)) for i in xrange(len(x))])
    ax.plot(x[:,0],x[:,1],np.zeros((len(x))),alpha=.7,lw=10)
    ax.quiver(0,0,0,0,0,1,pivot='tail',lw=5)
    return fig
