# 2016-08-11
from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# ---------------------- #
# Calculation functions. #
# ---------------------- #
def polar_angles(v0,a,b,c):
    """
    Get the polar and azimuthal angles that correspond to the new position of v0 after applying rotation 
    matrices with angles a,b,c.
    With YXZ.v0 convention.
    2016-08-12
    
    Params:
    -------
    v0 (floats)
        Starting vector.
    a,b,c (floats)
        Euler angles about x,y,z axes.
    """
    x,y,z = v0
    v1 = np.array([-z*np.cos(a)*np.sin(b) + y*(np.cos(c)*np.sin(a)*np.sin(b) + np.cos(b)*np.sin(c)) + 
                   x*(np.cos(b)*np.cos(c) - np.sin(a)*np.sin(b)*np.sin(c)),
                   y*np.cos(a)*np.cos(c) + z*np.sin(a) - x*np.cos(a)*np.sin(c),
                   z*np.cos(a)*np.cos(b) + x*(np.cos(c)*np.sin(b) + np.cos(b)*np.sin(a)*np.sin(c)) + 
                       y*(-np.cos(b)*np.cos(c)*np.sin(a) + np.sin(b)*np.sin(c))])
    
    phi = np.arctan2( v1[1],v1[0] )
    theta = np.arccos( v1[2] )
    return v1,phi,theta

def convert_euler_to_polar(yxz):
    """
    Convert Euler angles into polar/azimuthal angles.
    2016-08-12
    
    Params:
    -------
    yxz (ndarray)
        With columns for rotation angles about y,x,z axes.
    """
    if type(yxz) is pd.core.frame.DataFrame:
        yxz = yxz.values
    elif not type(yxz) is np.ndarray:
        raise Exception("Unexpected data type.")

    v = np.array([0,0,1])
    phis,thetas = np.zeros((len(yxz))),np.zeros((len(yxz)))
    for i,r in enumerate(yxz):
        v,phis[i],thetas[i] = polar_angles(v,r[1],r[0],r[2])
    return phis,thetas



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

def plot_euler_angles(bp):
    """
    Plot the 3D angle rotations.
    2016-08-11

    Params:
    -------
    bp (pd.DataFrame)
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    
    ax.plot(bp['x']/100,bp['z']/100,bp['y']/100,'o',alpha=.3)
    ax.set(xlim=[bp['x'].min()/100,bp['x'].max()/100],
           ylim=[bp['z'].min()/100,bp['z'].max()/100],
           zlim=[bp['y'].min()/100,bp['y'].max()/100])
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
