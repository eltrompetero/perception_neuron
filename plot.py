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
from numpy import sin,cos,pi
import pandas as pd
from ising.heisenberg import rotate
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline
import entropy.entropy as info
from scipy.signal import fftconvolve
from misc.utils import unique_rows
import data_access,utils



# ------------------- #
# Plotting functions. #
# ------------------- #
def shade_windows(vis,ax,t=None,
                  fill_kwargs={'color':'k','alpha':.2},
                  set_label=True):
    """
    Given axes, shade in the places where the avatar is invisible.

    Parameters
    ----------
    vis : ndarray
    ax : AxesSubplot
    t : ndarray,None
    fill_kwargs : dict,{}
    set_label : bool,True
    
    Returns
    -------
    None
    """
    if t is None:
        t = np.arange(len(vix))
    
    # Where invisible regions start.
    startix = np.where(np.diff(vis)==-1)[0]
    # and where they end
    endix = np.where(np.diff(vis)==1)[0]
    if endix[0]<startix[0]:
        startix = np.concatenate([[0],startix])
    if len(startix)>len(endix):
        endix = np.concatenate([endix,[len(vis)-1]])

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    for t0,t1 in zip(startix,endix):
        h = ax.fill_between([t[t0],t[t1]],*ylim,lw=0,**fill_kwargs)
    h.set_label('Invisible')
    ax.set(ylim=ylim,xlim=xlim)

def time_occlusion_trial(mbT,mbV,anT,anV,startEnd,visible,invisible,
                         fig=None,ax=None,
                         ylabel='Velocity (m/s)',
                         ylim=None,xlim=None):
    """
    Plot the temporal occlusion trial data.

    Params:
    -------
    mbT (ndarray of datetime)
        Model that is being tracked.
    mbV (ndarray)
    anT (ndarray of datetime)
        Axis Neuron calculation data of subject.
    anV (ndarray)
    visible,invisible (ndarray of datetime)
        Times at which tracked object becomes visible or invisible.
    """
    if fig is None:
        if ax is None:
            fig,ax = plt.subplots(figsize=(15,4))
        else:
            ax = fig.add_subplot(111)
    
    showIx = (mbT>startEnd[0]) & (mbT<startEnd[1])
    ax.plot(mbT[showIx],mbV[showIx],'b-')

    showIx = (anT>startEnd[0]) & (anT<startEnd[1])
    ax.plot(anT[showIx],anV[showIx],'r-')

    ylim = ylim or ax.get_ylim()
    for v,i in zip(visible,invisible):
        ax.fill_between([i,v],ylim[0],ylim[1],color='k',alpha=.2,lw=0)

    [l.set_rotation(90) for l in ax.xaxis.get_ticklabels()];
    ax.legend(('Model','Subject'),fontsize='x-small',loc=0)
    xlim = xlim or startEnd
    ax.set(ylabel=ylabel,xlim=xlim,ylim=ylim)
    return fig

def hist_dphase(delay,freq,ylim='low',laplace_counting=False):
    """
    Plot histogram of delay for given frequencies.

    Params:
    --------
    delay (ndarray)
        (n_freq,n_samples) Phase distance between two trajectories.
    freq (ndarray)
        Frequencies that are given.
    """
    from misc.plot import set_ticks_radian,colorcycle

    phaseLagPeaks = []
    fig,ax = plt.subplots(figsize=(7,4))
    c = colorcycle(len(freq))
    for freqix in range(len(freq)):
        n,x = np.histogram( delay[freqix],np.linspace(-pi,pi,30) )
        if laplace_counting:
            n += 1
        p = n/n.sum()
        
        ax.plot( x[:-1]+(x[1]-x[0])/2,n/n.sum(),'o-',alpha=1,c=c.next() )
        phaseLagPeaks.append( x[np.argmax(n)]+(x[1]-x[0])/2 )
    
    if ylim=='high':
        ylim = [0,1]
    elif ylim=='low':
        ylim = [0,.15]
    ax.set(xlim=[-pi,pi],xticks=[-pi,pi/2,0,pi/2,pi],ylim=ylim,
           xlabel='Phase lag',ylabel='Relative frequency',
           title='Histogram of phase lag')
    set_ticks_radian(ax,axis='x')
    ax.legend(freq,numpoints=1,title='Frequency',fontsize='small',
              bbox_to_anchor=[1.4,1.03])
    return fig,ax,phaseLagPeaks 

def cdf_dphase(delay,freq,title='Histogram of phase lag',xscale='f'):
    """
    Plot cdf of delay for given frequencies.

    Params:
    --------
    delay (ndarray)
        (n_freq,n_samples) Phase distance between two trajectories.
    freq (ndarray)
        Frequencies that are given.
    title (str)
    xscale (str)
        'f' means frequency scale and 't' means time scale
    """
    from misc.plot import set_ticks_radian,colorcycle
    from statsmodels.distributions import ECDF

    fig,ax = plt.subplots(figsize=(7,4))
    c = colorcycle(len(freq))
    for freqix in range(len(freq)):
        ecdf = ECDF( delay[freqix] )
        if xscale=='t':
            ax.plot( ecdf.x/(2*np.pi)/freq[freqix],ecdf.y,'-',alpha=1,c=c.next(),lw=2 )
        else:
            ax.plot( ecdf.x,ecdf.y,'-',alpha=1,c=c.next(),lw=2 )
    
    if xscale=='t':
        xlim = [-1/freq[0],1/freq[0]]
        xticks = np.arange(*xlim)
    else:
        xlim = [-pi,pi]
        xticks = [-pi,pi/2,0,pi/2,pi]
    ax.set(xlim=xlim,xticks=xticks,
           xlabel='Phase lag',ylabel='CDF',
           title=title)
    set_ticks_radian(ax,axis='x')
    ax.legend(['%1.1f Hz'%f for f in freq],numpoints=1,title='Frequency',fontsize='small',
              bbox_to_anchor=[1.4,1.03],labelspacing=.1)
    ax.grid()
    return fig,ax

def phase(T,v1,v2,phase,phasexyz,title='',maxshift=60,windowlength=100):
    """
    Plot normalized velocity phase lag graphs.

    Params:
    -------
    T,v1,v2
    phase
    phasexyz
    title
    maxshift (int=60)
    windowlength (int=100)
    """
    fig = plt.figure(figsize=(15,16))
    gs = gridspec.GridSpec(7,1)
    ax = [fig.add_subplot(gs[i]) for i in xrange(7)]

    # Velocity plots.
    h = []
    h.append( ax[0].plot(T,v1[:,0],'b-')[0] )
    h.append( ax[0].plot(T,v2[:,0],'r-')[0] )
    ax[0].set(ylabel=r'$v_x$',xticklabels=[],
              title=title)
    ax[0].legend(h,('Leader','Follower'),fontsize='xx-small',loc=0)
    ax[0].grid()

    ax[1].plot(T,v1[:,1],'b-')
    ax[1].plot(T,v2[:,1],'r-')
    ax[1].set(ylabel=r'$v_y$',xticklabels=[])
    ax[1].grid()

    ax[2].plot(T,v1[:,2],'b-')
    ax[2].plot(T,v2[:,2],'r-')
    ax[2].set(ylabel=r'$v_z$',xticklabels=[])
    ax[2].grid()

    # Phase lag plots.
    for i in xrange(3):
        ax[i+3].plot(T[maxshift:-maxshift-windowlength],phasexyz[i])
        ax[i+3].fill_between([T[0],T[-1]],-.25,.25,color='k',alpha=.1)
        ax[i+3].hlines(0,0,T[-1])
        ax[i+3].set(ylim=[-1,1],ylabel=['x phase','y phase','z phase'][i],xticklabels=[])
    ax[-1].plot(T[maxshift:-maxshift-windowlength],phase)
    ax[-1].set(ylim=[-1,1])
    ax[-1].fill_between([T[0],T[-1]],-.25,.25,color='k',alpha=.1)
    ax[-1].hlines(0,0,T[-1])

    # phasethreshold = .8
    # phase_ = phase.copy()
    # phase_[overlapcost<=phasethreshold] = nan
    # ax[4].plot(T[maxshift:-maxshift-windowlength],phase_,'b-',lw=2)
    # phase_ = phase.copy()
    # phase_[overlapcost>phasethreshold] = nan
    # ax[4].plot(T[maxshift:-maxshift-windowlength],phase_,'b-',alpha=.3)
    # # ax[4].fill_between(T[maxshift*2:-2*maxshift], phase-(1-overlapcost), phase+(1-overlapcost),
    # #                    color='b',alpha=.2)
    # smoothedPhase = fftconvolve(phase,ones((smoothingwindow))/smoothingwindow,mode='same')
    # ax[4].plot(T[maxshift:-maxshift-windowlength],
    #            smoothedPhase,'k-',lw=2,alpha=.5)
    # ax[4].fill_between([T[0],T[-1]],-.25,.25,color='k',alpha=.1)
    # ax[4].hlines(0,0,T[-1])
    # ax[4].set(xlabel='Time',ylabel='Phase lag',ylim=[-dt*maxshift,dt*maxshift])
    # ax[4].legend(('Raw phase lag','Moving avg','Below human rxn time'),loc=0,fontsize='xx-small')
    # ax[4].grid()

    for ax_ in ax:
        ax_.set(xlim=[T[0],T[-1]])

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
