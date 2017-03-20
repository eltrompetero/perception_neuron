# Functions for plotting Axis Neuron files.
# 
# Edward Lee edl56@cornell.edu
# 2017-03-10

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
import load,utils



# ------------------- #
# Plotting functions. #
# ------------------- #
def a_of_t(t,v1,v2,fig=None,ax=None):
    """
    Plot acceleration as a function of time for each axis independently to compare leader and follower.

    Params:
    -------
    t
    v1
    v2
    fig (None)
    ax (None)
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(15,8),sharex=True,sharey=True,nrows=3)
    
    for i in xrange(3):
        if i==2:
            ax[i].plot(t,v1[:,i]-1,'b-')
            ax[i].plot(t,v2[:,i]-1,'r-')
            ax[i].set(ylabel='Acc')
        else:
            ax[i].plot(t,v1[:,i],'b-')
            ax[i].plot(t,v2[:,i],'r-')
            ax[i].set(ylabel='Acc')
    ax[0].set(xlim=[t[0],t[-1]])
    ymx = max(np.abs(ax[0].get_ylim()))
    ax[-1].set(xlabel='Time (s)',ylim=[-ymx,ymx])
    fig.subplots_adjust(hspace=0)
    return fig,ax

def v_of_t(t,v1,v2,fig=None,ax=None):
    """
    Plot velocity as a function of time for each axis independently to compare leader and follower.

    Params:
    -------
    t
    v1
    v2
    fig (None)
    ax (None)
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(15,8),sharex=True,sharey=True,nrows=3)
    
    for i in xrange(3):
        ax[i].plot(t,v1[:,i],'b-')
        ax[i].plot(t,v2[:,i],'r-')
        ax[i].set(ylabel='Vel')
    ax[0].set(xlim=[t[0],t[-1]])
    ymx = max(np.abs(ax[0].get_ylim()))
    ax[-1].set(xlabel='Time (s)',ylim=[-ymx,ymx])
    fig.subplots_adjust(hspace=0)
    return fig,ax

def plot_xva_comparison(fig,ax,x1,x2,v1,v2,a1,a2,aOffset=0.,title=''):
    """
    Plot XVA comparison plots. Three rows of comparison.
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
    """
    Plot velocity and acceleration points against each other in two plots.
    """
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
