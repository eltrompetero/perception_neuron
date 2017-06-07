# Module for quick access to analysis pipelines.
# 2017-03-31
from __future__ import division
import pickle
import numpy as np
import os
from load import *
from utils import *
from filter import *

def extract_motionbuilder_model(trialno,person,modelhand):
    """
    Load model motion data. Assuming the play rate is a constant 1/60 Hz.
    
    Directory where animation data is stored is hard-coded.
    """
    from datetime import datetime,timedelta
    from workspace.utils import load_pickle

    dr = os.path.expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/Animations'
    fname = ['Eddie_%s_Hand_Model_19000_Recorded.p'%modelhand,
             'Eddie_%s_Hand_Model_Recorded.p'%modelhand,
             'Freya_(F)_Eddie_(L)_%s_Anim_Recorded.p'%modelhand][trialno]
    load_pickle('%s/%s'%(dr,fname))
    mbT = mbdf['Time'].values.astype(float)
    mbT -= mbT[0]
    mbV = savgol_filter( mbdf['%sHand'%modelhand].values,31,3,deriv=1,axis=0,delta=1/60 )/1000  # units of m/s

    # The time when the model starts is given in units of seconds. Convert to date time.
    dr = os.path.expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/%s'%person
    fname = '%s_visibility.txt'%['hand','arm','avatar'][trialno]
    visible,invisible = load_visibility(fname,dr)
    mbT = np.array([timedelta(seconds=t)+visible[0] for t in mbT])

    # Put these in the standard global coordinate system.
    mbV[:,:] = mbV[:,[1,0,2]]
    mbV[:,1] *= -1
    return mbT,mbV
    
def extract_AN_port(df,modelhand):
    """
    Take dataframe created from load_AN_port() and pull out the X, V, A data.
    """
    anT = array(map(datetime.utcfromtimestamp,df['Timestamp'].values.astype(datetime)/1e9))
    # anT = vectorize(datetime.utcfromtimestamp)(df['Timestamp'].values.astype(int)*1e-9)
    
    # Extract only necessary body part from the dataframe.
    df = load_calc('',cols='XVA',zd=False,df=df.ix[:,1:])
    if modelhand=='Left':
        _,anX,anV,anA = extract_calc_solo(leaderdf=df,bodyparts=['LeftHand'],dotruncate=0)
    else:
        _,anX,anV,anA = extract_calc_solo(leaderdf=df,bodyparts=['RightHand'],dotruncate=0)

    # Put these in the standard global coordinate system.
    for x,v,a in zip(anX,anV,anA):
        x[:,:] = x[:,[1,0,2]]
        x[:,2] *= -1
        v[:,:] = v[:,[1,0,2]]
        v[:,2] *= -1
        a[:,:] = a[:,[1,0,2]]
        a[:,2] *= -1
        
    return anT,anX,anV,anA

def quick_load(fileix,dt=1/120,negate_x=True,negate_y=False,disp=True):
    """
    Quick and dirty method for loading filtered velocities from hand trials. 

    Params:
    -------
    fileix
    dt (float=1/120)
    negate_x (bool=True)
    negate_y (bool=True)
    disp (bool=True
    
    Returns:
    --------
    T
    v1
    v2
    """
    bodypartix = hand_ix(fileix)
    fname = get_fnames()[fileix]
    if type(fname) is tuple:
        fname,date = fname
    else:
        date = None

    data = pickle.load(open('%s/%s.p'%(get_dr(fname,date),fname),'rb'))
    leaderX,leaderV,leaderA = data['leaderX'],data['leaderV'],data['leaderA']
    followerX,followerV,followerA = data['followerX'],data['followerV'],data['followerA']
    
    T = np.arange(len(leaderX[0]))*dt

    x1,x2 = leaderX[bodypartix],followerX[bodypartix]
    v1,v2 = leaderV[bodypartix],followerV[bodypartix]
    a1,a2 = leaderA[bodypartix],followerA[bodypartix]
    
    if negate_x:
        v2[:,0] *= -1
    if negate_y:
        v2[:,1] *= -1
    if disp:
        print np.corrcoef(v1[:,0],v2[:,0])[0,1]
        print np.corrcoef(v1[:,1],v2[:,1])[0,1]

    # Detrending necessary in some cases.
    if fileix in [51,52,53]:
        detrend(v2[:,2],inplace=True)
    return T,v1,v2

def pipeline_phase_calc(fileixs=[],
                        trajs=[],
                        suffix='',
                        down_sample=False,
                        **phase_calc_kwargs):
    """
    Pipeline loading pickled lowpass filtered data and running phase extraction after bandpass filtering.
    Pickles list of tuples phases, list of tuples vs (containing filtered velocities), and array fs
    (frequencies that were bandpassed).  Pickles are saved in phase_files/phase_%d.p

    If fileix's are given, those are loaded then processed. Otherwise, the data given is processed.
    
    Params:
    -------
    fileix (list)
        List of ints of files to load.
    trajs (list of tuples)
        (T,v1,v2) or ( T,(v1,v2,...,) )
    sample_freq (str='120')
        '120','60','30'
    bandwidth (float=.1)
        Bandwidth of bandpass filter.
    down_sample (bool=False)
        Down sample data by a factor of 2 if true.
    """
    if not os.path.isdir('phase_files'):
        os.makedirs('phase_files')
    #fs = np.concatenate((np.arange(-3,0,.1),np.arange(.1,3.1,.1)))
    fs = np.arange(.1,3.1,.1)
    
    if len(fileixs)>0:
        for fileix in fileixs:
            print "Starting file %d..."%fileix
            T,v1,v2 = quick_load(fileix,dt=1/int(sample_freq))
            if down_sample:
                T = T[::2]
                v1 = v1[::2]
                v2 = v2[::2]

            phases,vs = phase_calc(fs,v1,v2,**phase_calc_kwargs) 

            pickle.dump({'phases':phases,'vs':vs,'fs':fs},open('phase_files/phase_%d%s.p'%(fileix,suffix),'wb'),-1)
            print "Done with file %d."%fileix
    else:
        counter = 0
        for T,v in trajs:
            assert T.ndim==1 and all([i.ndim==1 for i in v])
            if down_sample:
                T = T[::2]
                v = [i[::2] for i in v]
            
            phases = []
            vs = []
            for i in v:
                phases_,vs_ = phase_calc(fs,i,**phase_calc_kwargs) 
                phases.append(phases_)
                vs.append(vs_)
            
            pickle.dump({'phases':phases,'vs':vs,'fs':fs},
                        open('phase_files/temp_phase_%d%s.p'%(counter,suffix),'wb'),-1)
            print "Done with file %d."%counter
            counter += 1

def phase_calc(fs,v1,v2=None,
               sample_freq='120',
               bandwidth=.1,
               down_sample=False):
    """
    Params:
    -------
    fs (ndarray)
        Frequencies at which to bandpass.
    v1 (ndarray)
        1d arrays.
    v2 (ndarray=None)
    sample_freq (str='120')
        '120' or '60'
    bandwidth (float=.1)
        Bandwidth of bandpass filter.
    down_sample (bool=False)
        Down sample data by a factor of 2 if true.
    """
    from scipy.signal import hilbert

    if str(sample_freq)=='120':
        windowLength,filtwidth = 501,50
    elif str(sample_freq=='60'):
        windowLength,filtwidth = 251,25
    elif str(sample_freq=='30'):
        windowLength,filtwidth = 125,13
    else:
        raise NotImplementedError
    
    phases = []
    vs = []
    for f in fs:
        v1_ = moving_freq_filt(v1,window=windowLength,
                               window_type=('gaussian',filtwidth),
                               filter_type='single',
                               sample_freq=int(sample_freq),
                               pass_freq=f,
                               bandwidth=bandwidth,
                               axis=0)
        h1 = hilbert(v1_,axis=0)
        phase1 = np.angle(h1)

        if not v2 is None:
            v2_ = moving_freq_filt(v2,window=windowLength,
                                   window_type=('gaussian',filtwidth),
                                   filter_type='single',
                                   sample_freq=int(sample_freq),
                                   pass_freq=f,
                                   bandwidth=bandwidth,
                                   axis=0)
            h2 = hilbert(v2_,axis=0)
        
            phase2 = np.angle(h2)
            phases.append((phase1,phase2))
            vs.append((v1_,v2_))
        else:
            phases.append(phase1)
            vs.append(v1_)
    
    return phases,vs

def filter_hand_trials(filesToFilter,dt=1/60,
        extract_calc_kwargs={'rotate_to_face':False,
                             'remove_hip_drift':True,
                             'dotruncate':5},
        filterparams='default',
        suffix=''):
    """
    Shortcut for filtering hand trials data by just giving file number.
    2017-03-19
    
    Params:
    -------
    filesToFilter (list)
    dt (float=1/60)
    extract_calc_kwargs (dict)
    filterparams (str='default')
        Choose between 'default' and '120'. Filter parameters for butterworth filter as in utils.smooth()
    """
    from filter import smooth
    import cPickle as pickle
    
    bodyparts = [['RightHand','LeftHand'],
                 ['LeftHand','RightHand']]

    for fileix in filesToFilter:
        # Read position, velocity and acceleration data from files.
        fname = get_fnames()[fileix]
        if type(fname) is tuple:
            fname,date = fname
        else:
            date = None
        T,leaderX,leaderV,leaderA,followerX,followerV,followerA = extract_calc(fname,
                                                                   get_dr(fname,date),
                                                                   bodyparts,
                                                                   dt,
                                                                   rotation_angle=global_rotation(fileix),
                                                                   **extract_calc_kwargs)

        for x in leaderX:
            x-=x.mean(0)
        for x in followerX:
            x-=x.mean(0)
        
        # Butterworth filter data and pickle it.
        for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:] = smooth(x,filterparams=filterparams)[:]
            v[:] = smooth(v,filterparams=filterparams)[:]
            a[:] = smooth(a,filterparams=filterparams)[:]
        for x,v,a in zip(followerX,followerV,followerA):
            x[:] = smooth(x,filterparams=filterparams)[:]
            v[:] = smooth(v,filterparams=filterparams)[:]
            a[:] = smooth(a,filterparams=filterparams)[:]
        
        # Save into same directory as calc file.
        savedr = '%s/%s%s.p'%(get_dr(fname,date),fname,suffix)
        print "Saving as %s"%savedr
        pickle.dump({'T':T,
                     'leaderX':leaderX,'followerX':followerX,
                     'leaderV':leaderV,'followerV':followerV,
                     'leaderA':leaderA,'followerA':followerA},
                    open(savedr,'wb'),-1)

