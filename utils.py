# 2016-08-11
from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin,cos
import pandas as pd
from ising.heisenberg import rotate
from scipy.interpolate import LSQUnivariateSpline
import entropy.entropy as info
from scipy.signal import fftconvolve

# ---------------------- #
# Calculation functions. #
# ---------------------- #
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
    return fftconvolve(x,np.ones((filtDuration)),mode='same')

def discrete_vel(v,timescale):
    """
    Smooth velocity with moving average, sample with frequency inversely proportional to width of moving
    average and take the sign of the diff.
    2016-11-09

    Params:
    -------
    v (vector)
    timescale (int)
    """
    vsmooth = moving_mean_smooth(v,timescale)
    vsmooth = vsmooth[::timescale]
    
    change = np.sign(np.diff(vsmooth))
    return change

def convert_t_to_bins(timescale,dt):
    """
    Convert timescale given in units of seconds to bins for a moving average in the given sample. Remove
    duplicate timescale entries that appear when this discretization to bins has occurred.
    2016-11-07

    Params:
    -------
    timescale (int)
    dt (float)
    """
    discretetimescale = (timescale/dt).astype(int)
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

def spline_smooth(t,Y):
    """
    Use quintic least squares spline to smooth given data in place down the columns. Knots appear about every second as estimated from the inverse sampling rate.
    2016-10-30

    Params:
    -------
    t (vector)
        Time of measurements
    Y (ndarray) 
        n_time x n_dim. Measurements.

    Value:
    ------
    spline (list)
        List of LSQUnivariateSpline instances for each col of input Y.
    """
    dt = t[1]-t[0]
    T = int(1/dt)
    spline = []
    for i,y in enumerate(Y.T):
        spline.append( LSQUnivariateSpline(t,y,t[T::T],k=5) )
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




# ------------------- #
# Plotting functions. #
# ------------------- #
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
