# Module for loading and extracting data from Axis Neuron files.
#
# Classes:
# Node, Tree
# 
# Author: Edward D. Lee
# Email: edl56@cornell.edu
# 2017-03-28

from __future__ import division
import pandas as pd
import numpy as np
from utils import *
import cPickle as pickle
import os

def get_fnames():
    return ['Eddie L Caeli F March',
          'Eddie F Caeli L March',
          'Eddie J Caeli J Tango',
          'Eddie F Caeli L Tango',
          'Eddie L Caeli F Tango',
          'Caeli (L) Vincent (F) March',
          'Caeli (F) Vincent (L) March',
          'Caeli (L) Vincent (F) Tango',
          'Caeli (F) Vincent (L) Tango',
          'Caeli (J) Vincent (J) Tango',
          'Caeli (L) Vincent (F) Hands',
          'Caeli (F) Vincent (L) Hands',
          'Caeli (J) Vincent (J) Hands',
          'Itai (L) Anja (F) March',
          'Itai (F) Anja (L) March',
          'Itai (L) Anja (F) Tango',
          'Itai (F) Anja (L) Tango',
          'Itai (J) Anja (J) Tango',
          'Itai (L) Anja (F) Hands',
          'Itai (F) Anja (L) Hands',
          'Itai (J) Anja (J) Hands',
          'Itai (J) Anja (J) Hands_1',
          'Caeli (J) Eddie (J) Hands Cal',
          'Caeli (L) Eddie (F) Hands',
          'Caeli (F) Eddie (L) Hands',
          'Caeli (J) Eddie (J) Hands',
          'Caeli (J) Eddie (J) Hands Cal After',
          'Caeli (J) Eddie (J) Hands Cal1 Blind',
          'Caeli (L) Eddie (F) Hands Blind',
          'Caeli (F) Eddie (L) Hands Blind',
          'Caeli (J) Eddie (J) Hands Cal2 Blind',
          'Caeli (J) Sam (J) Hands Cal1',
          'Caeli (L) Sam (F) Hands',
          'Caeli (F) Sam (L) Hands',
          'Caeli (J) Sam (J) Hands Cal2',
          'Caeli (J) Sam (J) Hands',
          'Caeli (J) Eddie (J) Fine Cal 1',
          'Caeli (J) Eddie (J) Fine Cal 2',
          'Caeli (J) Eddie (J) Fine',
          'Caeli (J) Eddie (J) Fine Cal 3',
          'Caeli (J) Eddie (J) Fine Cal 4',
          'Caeli (J) Yunus (J) Cal 1',
          'Caeli (J) Yunus (J) Cal 2',
          'Caeli (L) Yunus (F)',
          'Caeli (J) Yunus (J) Cal 3',
          'Caeli (J) Yunus (J) Cal 4',
          'Caeli (F) Yunus (L)',
          'Caeli (J) Yunus (J) Cal 5',
          'Caeli (J) Yunus (J) Cal 6',
          'Caeli (J) Yunus (J)',
          'Caeli (J) Yunus (J) Cal 7',
          ('Caeli (J) Eddie (J) Half Occlusion','20170307'),
          ('Caeli (J) Eddie (J) Full Occlusion','20170307'),
          ('Caeli (J) Eddie (J) Low Light','20170307'),
          ('Caeli (L) Eddie (F) Hands Startup Timer','20170310'),
          ('Caeli (F) Eddie (L) Hands Startup Timer','20170310'),
          ('Caeli (L) Eddie (F) Hands','20170317'),
          ('Caeli (F) Eddie (L) Hands','20170317'),
          ('Caeli (J) Eddie (J) Hands','20170317'),
          ('Caeli (J) Eddie (J) Hands Half Occlusion','20170317'),
          ('Caeli (J) Eddie (J) Hands Full Occlusion','20170317'),
          ('Caeli (J) Eddie (J) Hands Low Light','20170317'),
          ('Caeli (L) Eddie (F)','20170418'),
          ('Caeli (F) Eddie (L)','20170418'),
          ('Caeli (J) Eddie (J)','20170418'),
          ('Caeli (J) Eddie (J) Right Left Eye Closed','20170418'),
          ('Caeli (J) Eddie (J) Left Right Eye Closed','20170418'),
          ('Caeli (J) Eddie (J) Half Occlusion','20170418'),
          ('Henry (L) Winnie (F)','20170420'),
          ('Henry (F) Winnie (L)','20170420'),
          ('Henry (J) Winnie (J)','20170420'),
          ('Henry (J) Winnie (J) 2','20170420'),
          ('Henry (J) Winnie (J) Low Light','20170420'),
          ('Eddie (L) Freya (F)','20170424'),
          ('Eddie (F) Freya (L)','20170424'),
          ('Eddie (J) Freya (J)','20170424'),
          ('Eddie (J) Freya (J) Low Light','20170424')
          ]

def hand_ix(fileix):
    """
    Return the hand that was used for the trial given the trial number. This is returned as the index to
    bodyparts defined as 
    bodyparts  = [['RightHand','LeftHand'],
                  ['LeftHand','RightHand']]

    Params:
    -------
    fileix (int)
    """
    if type(fileix) is int:
        fileix = str(fileix)
    
    return {'18':1,
            '19':1,
            '20':1,
            '21':1,
            '23':0,
            '24':1,
            '32':0,
            '33':1,
            '35':0,
            '43':0,
            '46':1,
            '49':1,
            '51':1,
            '52':1,
            '53':1,
            '56':0,
            '57':1,
            '58':1,
            '59':1,
            '60':0,
            '61':1,
            '62':1,
            '63':0,
            '64':0,
            '65':1,
            '68':0,
            '69':1,
            '70':1,
            '71':0,
            '72':1,
            '73':0,
            '74':1,
            '75':1}.get(fileix,None)

def global_rotation(fileix):
    """
    Return the angle that the individuals should be rotated by such that they are facing each other across the
    x-axis.

    Params:
    -------
    fileix (int)
    """
    if type(fileix) is int:
        fileix = str(fileix)
    
    return {'43':np.pi/2,
            '46':np.pi/2,
            '49':np.pi/2}.get(fileix,0)


def get_dr(fname,date=None):
    """Return directory where files are saved."""
    from os.path import expanduser
    homedr = expanduser('~')
    datadr = 'Dropbox/Documents/Noitom/Axis Neuron/Motion Files'

    if not date is None:
        return {'20170307':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie_Occlusion'),
                '20170310':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie_Startup'),
                '20170317':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie'),
                '20170418':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie'),
                '20170420':'%s/%s/%s_%s'%(homedr,datadr,date,'Henry_Winnie'),
                '20170424':'%s/%s/%s_%s'%(homedr,datadr,date,'Eddie_Freya')}[date]

    if 'Itai' in fname and 'Anja' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20161205_Itai_Anja/'
    elif 'Caeli' in fname and 'Vincent' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20161130_Caeli_Vincent/'
    elif 'Caeli' in fname and 'Eddie' in fname and 'Startup' in fname:
        return (expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion '+
                'Files/20170310_Caeli_Eddie_Startup/')
    elif 'Caeli' in fname and 'Eddie' in fname and ('Occlusion' in fname or 'Low' in fname):
        return ( expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion '+
                 'Files/20170307_Caeli_Eddie_Occlusion/' )
    elif 'Caeli' in fname and 'Eddie' in fname and 'Blind' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170127_Caeli_Eddie/'
    elif 'Caeli' in fname and 'Eddie' in fname and not 'Fine' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170124_Caeli_Eddie/'
    elif 'Caeli' in fname and 'Sam' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170127_Caeli_Sam/'
    elif 'Caeli' in fname and 'Eddie' in fname and 'Fine' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170202_Caeli_Eddie/'
    elif 'Caeli' in fname and 'Yunus' in fname:
        return expanduser('~')+'/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/20170203_Caeli_Yunus/'
    else:
        raise Exception("Invalid file name.")

def print_files(ix0=0,ix1=None):
    """
    Print list of available file names with their indices to make it easy to load them.
    2017-01-18

    Params:
    -------
    ix0 (int=0)
    ix1 (int=None)
    """
    fnames=get_fnames()
    ix1 = ix1 or len(fnames)
    fnames = fnames[ix0:ix1]
    for i,f in enumerate(fnames):
        print "%d\t%s"%(i+ix0,f)

def calc_file_body_parts():
    """
    According to ref file sent by Noitom.
    2016-11-13
    """
    return ['Hips',
            'RightUpLeg',
            'RightLeg',
            'RightFoot',
            'LeftUpLeg',
            'LeftLeg',
            'LeftFoot',
            'RightShoulder',
            'RightArm',
            'RightForeArm',
            'RightHand',
            'LeftShoulder',
            'LeftArm',
            'LeftForeArm',
            'LeftHand',
            'Head',
            'Neck',
            'Spine3',
            'Spine2',
            'Spine1',
            'Spine',
            'left foot contact',
            'right foot contact',
            'RightHandThumb1',
            'RightHandThumb2',
            'RightHandThumb3',
            'RightInHandIndex',
            'RightHandIndex1',
            'RightHandIndex2',
            'RightHandIndex3',
            'RightInHandMiddle',
            'RightHandMiddle1',
            'RightHandMiddle2',
            'RightHandMiddle3',
            'RightInHandRing',
            'RightHandRing1',
            'RightHandRing2',
            'RightHandRing3',
            'RightInHandPinky',
            'RightHandPinky1',
            'RightHandPinky2',
            'RightHandPinky3',
            'LeftHandThumb1',
            'LeftHandThumb2',
            'LeftHandThumb3',
            'LeftInHandIndex',
            'LeftHandIndex1',
            'LeftHandIndex2',
            'LeftHandIndex3',
            'LeftInHandMiddle',
            'LeftHandMiddle1',
            'LeftHandMiddle2',
            'LeftHandMiddle3',
            'LeftInHandRing',
            'LeftHandRing1',
            'LeftHandRing2',
            'LeftHandRing3',
            'LeftInHandPinky',
            'LeftHandPinky1',
            'LeftHandPinky2',
            'LeftHandPinky3']

def calc_file_headers():
    """Load calc file headers from pickle."""
    import os,pickle
    headers = pickle.load(open('%s/%s'%(os.path.expanduser('~'),
                          'Dropbox/Research/py_lib/perceptionneuron/calc_file_headers.p'),'rb'))['headers']
    return headers

def subject_settings_v3(index,return_list=True):
    settings = [{'person':'Zimu3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Darshna3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Richard3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Rachel3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Priyanka3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Emily3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Sam3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Najila3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Kemper3',
                  'modelhandedness':['Left','Right','Left','Right'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']},
                {'person':'Lauren3',
                  'modelhandedness':['Right','Left','Right','Left'],
                  'rotation':[0,0,0,0],
                  'trials':['avatar0','avatar1','hand0','hand1']}
                ][index]
    dr = (os.path.expanduser('~')+
          '/Dropbox/Documents/Noitom/Axis Neuron/Motion Files/UE4_Experiments/%s'%settings['person'])
    if return_list:
        output = [settings[k] for k in ['person','modelhandedness','rotation']]
        output.append(dr)
        return output
    return settings,dr

def load_calc(fname,cols='V',read_csv_kwargs={},zd=True,df=None):
    """
    Load calculation file output by Axis Neuron. 
    Note that z-axis points into the ground by default.
    2016-12-05

    Params:
    -------
    fname (str)
    skeleton (list of str)
        Names fo the bones specified in fname.
    cols (str)
        Data columns to keep. Columns are XVQAW (position, vel, quaternion, acc, angular vel)
    """
    from ising.heisenberg import rotate
    
    if df is None:
        df = pd.read_csv(fname,skiprows=5,sep='\t',**read_csv_kwargs)
    
    # Only keep desired columns.
    keepix = np.zeros((len(df.columns)),dtype=bool)
    for s in cols:
        keepix += np.array([s in c for c in df.columns])
    df = df.iloc[:,keepix]
    columns = list(df.columns)

    # Rename numbered columns by body parts.
    skeleton = calc_file_body_parts()
    nameIx = 0
    for i,s in enumerate(skeleton):
        if not 'contact' in s:
            for j,c in enumerate(columns):
                columns[j] = c.replace(str(nameIx+1).zfill(2),s)
            nameIx += 1
    df.columns = columns
    
    # Read Zd axis, the original direction that the wearer is facing.
    if zd:
        with open(fname,'r') as f:
            zd = np.array([float(i) for i in f.readline().split('\t')[1:]])
        return df,zd
    #n = np.cross(zd,np.array([-1,0,0]))
    #theta = np.arccos(zd.dot([-1,0,0]))
    #for i in xrange(len(df.columns)):
    #    if any([c+'-x' in df.columns[i] for c in cols]):
    #        df.iloc[:,i:i+3].values[:,:] = rotate(df.iloc[:,i:i+3].values,n,theta)
    return df

def extract_parts(df,bodyparts):
    """
    Extract columns for specific body parts from loaded calc file by comparing with column headers.
    
    Params:
    -------
    df (pd.DataFrame)
    bodyparts (list)
    """
    bodyparts = [s.lower() for s in bodyparts]
    returnix = []
    columnBodyParts = [c.split('-')[0].lower() for c in df.columns]
    for i,c in enumerate(columnBodyParts):
        if c in bodyparts:
            returnix.append(i)
    return df.iloc[:,returnix]

def group_cols(columns):
    """
    Group columns of 3 into multiindex with xyz subcolumns.
    2017-03-03
    """
    bodyparts = [c.split('-')[0] for c in columns[::3]]
    return pd.MultiIndex.from_product((bodyparts,['x','y','z'])) 

def extract_calc_solo(fname='',dr='',bodyparts=[],dt=1/120,
                      leaderdf=None,
                      append=True,
                      dotruncate=5,
                      remove_hip_drift=True,
                      usezd=False,
                      read_csv_kwargs={},
                      center_x=False,
                      rotation_angle=False
                      ):
    """
    Extract specific set of body parts from calculation file with one individual. This is modification of
    extract_calc(). 

    Params:
    -------
    fname (str)
    dr (str)
    bodyparts (list of strings)
        Body parts to keep.
    dt (float)
    leaderdf (pandas.DataFrame=None)
        If given, this will be the data array used to extract data.
    append (bool=True)
        If true, keep list of data from bodyparts else add all the velocities and acceleration together. This
        is useful if we're looking at the motion of the feet and want to look at the sum of the motion of the
        feet (because we don't care about stationary feet).
    dotruncate (float=5)
        Truncate beginning and end of data by this many seconds.
    remove_hip_drift (bool=True)
    usezd (bool=True)
        Get initial body orientation from calc file's Zd entry. This seems to not work as well in capturing
        the 3 dimension of hand movement. I'm not sure why, but I would assume because the orientation between
        hands and the body is not totally accurate according to Axis Neuron.
    read_csv_kwargs (dict)
        Passed onto pandas.read_csv
    center_x (bool=False)
        Subtract mean from the mean of each body parts' displacement.
    rotation_angle (int=False)
        If an integer or float, X, V, A will be rotated about the local [0,0,1] z-axis. Note that this z-axis
        points into the ground.

    Returns:
    --------
    T,X,V,A
    """
    from ising.heisenberg import rotate
    skeleton = calc_file_body_parts()
    
    if leaderdf is None:
        # Read position, velocity and acceleration data from files.
        print "Loading file %s"%fname
        leaderdf,leaderzd = load_calc('%s/%s.calc'%(dr,fname),
                                      cols='XVA',
                                      read_csv_kwargs=read_csv_kwargs)
            
    T = np.arange(len(leaderdf))*dt

    if remove_hip_drift:
        # Remove drift in hips.
        Xix = np.array(['X' in c for c in leaderdf.columns])
        leaderdf.iloc[:,Xix] -= np.tile(leaderdf.iloc[:,:3],(1,leaderdf.shape[1]//9))

    # Select out the body parts that we want.
    bodypartix = [skeleton.index(b) for b in bodyparts]
    
    if append:
        leaderX,leaderV,leaderA = [],[],[]
        for i,iloc in enumerate(bodypartix):
            leaderX.append( leaderdf.values[:,iloc*9:iloc*9+3].copy() ) 
            leaderV.append( leaderdf.values[:,iloc*9+3:iloc*9+6].copy() ) 
            leaderA.append( leaderdf.values[:,iloc*9+6:iloc*9+9].copy() ) 
    else:
        for i,iloc in enumerate(bodypartix):
            if i==0:
                leaderX = [leaderdf.values[:,iloc*9:iloc*9+3].copy()]
                leaderV = [leaderdf.values[:,iloc*9+3:iloc*9+6].copy()]
                leaderA = [leaderdf.values[:,iloc*9+6:iloc*9+9].copy()]
            else:
                leaderV[0] += leaderdf.values[:,iloc*9+3:iloc*9+6]
                leaderA[0] += leaderdf.values[:,iloc*9+6:iloc*9+9]
                
    if rotation_angle:
        for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:,:] = rotate(x,np.array([0,0,1.]),rotation_angle)
            v[:,:] = rotate(v,np.array([0,0,1.]),rotation_angle)
            a[:,:] = rotate(a,np.array([0,0,1.]),rotation_angle)

    # Truncate beginning and ends of data set.
    if dotruncate:
        if not type(dotruncate) is list:
            dotruncate = [dotruncate]*2
        
        counter=0
        for x,v,a in zip(leaderX,leaderV,leaderA):
            leaderX[counter] = truncate(T,x,t0=dotruncate[0],t1=dotruncate[1])
            leaderV[counter] = truncate(T,v,t0=dotruncate[0],t1=dotruncate[1])
            leaderA[counter] = truncate(T,a,t0=dotruncate[0],t1=dotruncate[1])
            counter += 1
        T = truncate(T,T,t0=dotruncate[0],t1=dotruncate[1])

    if center_x:
        for x in leaderX:
            x -= x.mean(0)

    return T,leaderX,leaderV,leaderA

def extract_calc(fname,dr,bodyparts,dt,
                 append=True,
                 dotruncate=5,
                 remove_hip_drift=True,
                 rotate_to_face=False,
                 usezd=False,
                 read_csv_kwargs={},
                 center_x=False,
                 rotation_angle=False
                ):
    """
    Extract specific set of body parts from calculation file with two individuals. If a file with
    coordination of hands is given, then I have to align the subjects to a global coordinate frame defined by
    the initial orientation of their hands.

    For import of hands trials, the first axis is the direction along which the subjects are aligned.

    The slowest part is loading the data from file.

    Params:
    -------
    fname (str)
    dr (str)
    bodyparts (list of list of strings)
        First list is for leader and second list is for follower.
    dt (float)
    append (bool=True)
        If true, keep list of data from bodyparts else add all the velocities and acceleration together. This
        is useful if we're looking at the motion of the feet and want to look at the sum of the motion of the
        feet (because we don't care about stationary feet).
    dotruncate (float=5)
        Truncate beginning and end of data by this many seconds.
    remove_hip_drift (bool=True)
    rotate_to_face (bool=True)
        Rotate the individuals to face each other.
    usezd (bool=True)
        Get initial body orientation from calc file's Zd entry. This seems to not work as well in capturing
        the 3 dimension of hand movement. I'm not sure why, but I would assume because the orientation between
        hands and the body is not totally accurate according to Axis Neuron.
    read_csv_kwargs (dict)
        Passed onto pandas.read_csv
    center_x (bool=False)
        Subtract mean from the mean of each body parts' displacement.
    rotation_angle (float=False)
        If a scalar, both individuals will rotated by that many radians about the origin. Useful for trials
        where individuals were set up facing a different direction in trials than in initial calibration.

    Value:
    ------
    T,leaderX,leaderV,leaderA,followerX,followerV,followerA
    """
    skeleton = calc_file_body_parts()

    # Read position, velocity and acceleration data from files.
    print "Loading file %s"%fname
    leaderix = 1 if ('F' in fname.split(' ')[1]) else 0
    if not 'leaderdf' in globals():
        characters = [fname.split(' ')[0],fname.split(' ')[2]]
        leaderdf,leaderzd = load_calc('%s/%s%s.calc'%(dr,fname,characters[leaderix]),
                                      cols='XVA',
                                      read_csv_kwargs=read_csv_kwargs)
        followerdf,followerzd = load_calc('%s/%s%s.calc'%(dr,fname,characters[1-leaderix]),
                                          cols='XVA',
                                          read_csv_kwargs=read_csv_kwargs)
        
        T = np.arange(len(followerdf))*dt

    if remove_hip_drift:
        # Remove drift in hips.
        Xix = np.array(['X' in c for c in leaderdf.columns])
        leaderdf.iloc[:,Xix] -= np.tile(leaderdf.iloc[:,:3],(1,leaderdf.shape[1]//9))
        followerdf.iloc[:,Xix] -= np.tile(followerdf.iloc[:,:3],(1,followerdf.shape[1]//9))

    if rotate_to_face:
        # The correction for the hands is a bit involved. First, I remove the drift from the hips from all the
        # position of all body parts. Then, take the hands, and center them by their midpoint. Then I rotate
        # their positions so that the leader and follower are facing each other. Remember that the z-axis is
        # pointing into the ground!
        assert remove_hip_drift, "Must remove hip drift if we have a hands trial."
        
        # Use initial condition to set orientation of subjects, 
        if usezd:
            bodyvec=[leaderzd,followerzd]
        else:
            from utils import initial_orientation
            bodyvec = [initial_orientation(leaderdf),initial_orientation(followerdf)]

    # Select out the body parts that we want.
    bodypartix = [[skeleton.index(b) for b in bodyparts_] 
                   for bodyparts_ in bodyparts]
    
    if append:
        leaderX,leaderV,leaderA,followerX,followerV,followerA = [],[],[],[],[],[]
        for i,iloc in enumerate(bodypartix[leaderix]):
            leaderX.append( leaderdf.values[:,iloc*9:iloc*9+3].copy() ) 
            leaderV.append( leaderdf.values[:,iloc*9+3:iloc*9+6].copy() ) 
            leaderA.append( leaderdf.values[:,iloc*9+6:iloc*9+9].copy() ) 
        for i,iloc in enumerate(bodypartix[1-leaderix]):
            followerX.append( followerdf.values[:,iloc*9:iloc*9+3].copy() )
            followerV.append( followerdf.values[:,iloc*9+3:iloc*9+6].copy() )
            followerA.append( followerdf.values[:,iloc*9+6:iloc*9+9].copy() )
    else:
        for i,iloc in enumerate(bodypartix[leaderix]):
            if i==0:
                leaderX = [leaderdf.values[:,iloc*9:iloc*9+3].copy()]
                leaderV = [leaderdf.values[:,iloc*9+3:iloc*9+6].copy()]
                leaderA = [leaderdf.values[:,iloc*9+6:iloc*9+9].copy()]
            else:
                leaderV[0] += leaderdf.values[:,iloc*9+3:iloc*9+6]
                leaderA[0] += leaderdf.values[:,iloc*9+6:iloc*9+9]
                
        for i,iloc in enumerate(bodypartix[1-leaderix]):
            if i==0:
                followerX = [followerdf.values[:,iloc*9:iloc*9+3].copy()]
                followerV = [followerdf.values[:,iloc*9+3:iloc*9+6].copy()]
                followerA = [followerdf.values[:,iloc*9+6:iloc*9+9].copy()]
            else:
                followerV[0] += followerdf.values[:,iloc*9+3:iloc*9+6]
                followerA[0] += followerdf.values[:,iloc*9+6:iloc*9+9]

    if rotate_to_face:
        # Make sure that first dimension corresponds to the axis towards the other person.
        # Make the leader face towards ([1,0,0]) and the follower towards [-1,0,0].
        phi = []
        phi.append(np.arccos(bodyvec[0][0]) if bodyvec[0][1]<0 else -np.arccos(bodyvec[0][0]))
        phi.append(np.arccos(-bodyvec[1][0]) if (-bodyvec[1][1])<0 else -np.arccos(-bodyvec[1][0]))
        
        for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:,:] = rotate(x,np.array([0,0,1.]),phi[0])
            v[:,:] = rotate(v,np.array([0,0,1.]),phi[0])
            a[:,:] = rotate(a,np.array([0,0,1.]),phi[0])
        for x,v,a in zip(followerX,followerV,followerA):
            x[:,:] = rotate(x,np.array([0,0,1.]),phi[1])
            v[:,:] = rotate(v,np.array([0,0,1.]),phi[1])
            a[:,:] = rotate(a,np.array([0,0,1.]),phi[1])

        for i,s in enumerate(bodyparts[leaderix]):
            if 'Right' in s or 'Left' in s:
                # Shift subjects away from each other so that they're actually facing each over a gap.
                leaderX[i][:,0] -= 1
        for i,s in enumerate(bodyparts[1-leaderix]):
            if 'Right' in s or 'Left' in s:
                # Shift subjects away from each other so that they're actually facing each over a gap.
                followerX[i][:,0] += 1
            
                # Reflect follower about mirror over the axis parallel to the "mirror."
                followerV[i][:,0] *= -1
                followerA[i][:,0] *= -1
    elif rotation_angle:
        for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:,:] = rotate(x,np.array([0,0,1.]),rotation_angle)
            v[:,:] = rotate(v,np.array([0,0,1.]),rotation_angle)
            a[:,:] = rotate(a,np.array([0,0,1.]),rotation_angle)
        for x,v,a in zip(followerX,followerV,followerA):
            x[:,:] = rotate(x,np.array([0,0,1.]),rotation_angle)
            v[:,:] = rotate(v,np.array([0,0,1.]),rotation_angle)
            a[:,:] = rotate(a,np.array([0,0,1.]),rotation_angle)


    # Truncate beginning and ends of data set.
    if dotruncate:
        if not type(dotruncate) is list:
            dotruncate = [dotruncate]*2
        
        counter=0
        for x,v,a in zip(leaderX,leaderV,leaderA):
            leaderX[counter] = truncate(T,x,t0=dotruncate[0],t1=dotruncate[1])
            leaderV[counter] = truncate(T,v,t0=dotruncate[0],t1=dotruncate[1])
            leaderA[counter] = truncate(T,a,t0=dotruncate[0],t1=dotruncate[1])
            counter += 1
        counter=0
        for x,v,a in zip(followerX,followerV,followerA):
            followerX[counter] = truncate(T,x,t0=dotruncate[0],t1=dotruncate[1])
            followerV[counter] = truncate(T,v,t0=dotruncate[0],t1=dotruncate[1])
            followerA[counter] = truncate(T,a,t0=dotruncate[0],t1=dotruncate[1])
            counter += 1
        T = truncate(T,T,t0=dotruncate[0],t1=dotruncate[1])

    if center_x:
        for x in leaderX:
            x -= x.mean(0)
        for x in followerX:
            x -= x.mean(0)

    return T,leaderX,leaderV,leaderA,followerX,followerV,followerA

def extract_W(fname,dr,bodyparts,dt,
                 dotruncate=5,
                ):
    """
    Extract angular velocities of specified set of body parts from calculation file without any adjustment.

    The slowest part is loading the data from file.
    2017-01-28

    Params:
    -------
    fname (str)
    dr (str)
    bodyparts (list of list of strings)
        First list is for leader and second list is for follower.
    dt (float)
    dotruncate (float=5)
        Truncate beginning and end of data by this many seconds.

    Value:
    ------
    T,leaderW,followerW
    """
    skeleton = calc_file_body_parts()

    # Read position, velocity and acceleration data from files.
    leaderix = 1 if ('F' in fname.split(' ')[1]) else 0
    if not 'leaderdf' in globals():
        characters = [fname.split(' ')[0],fname.split(' ')[2]]
        leaderdf,leaderzd = load_calc('%s%s%s.calc'%(dr,fname,characters[leaderix]),cols='W')
        followerdf,followerzd = load_calc('%s%s%s.calc'%(dr,fname,characters[1-leaderix]),cols='W')
        
        T = np.arange(len(followerdf))*dt

    # Select out the body parts that we want.
    bodypartix = [[skeleton.index(b) for b in bodyparts_] 
                   for bodyparts_ in bodyparts]
    
    leaderW,followerW = [],[]
    for i,iloc in enumerate(bodypartix[leaderix]):
        leaderW.append( leaderdf.values[:,iloc*9:iloc*9+3].copy() ) 
    for i,iloc in enumerate(bodypartix[1-leaderix]):
        followerW.append( followerdf.values[:,iloc*9:iloc*9+3].copy() )
    
    # Truncate beginning and ends of data set.
    if dotruncate:
        counter=0
        for w in leaderW:
            leaderW[counter] = truncate(T,w,t0=dotruncate,t1=dotruncate)
            counter += 1
        counter=0
        for w in followerW:
            followerW[counter] = truncate(T,w,t0=dotruncate,t1=dotruncate)
            counter += 1
        T = truncate(T,T,t0=dotruncate,t1=dotruncate)
    return T,leaderW,followerW

def load_bvh(fname,includeDisplacement=False,removeBlank=True):
    """
    Load data from BVH file. Euler angles are given as YXZ. Axis Neuron only keeps track of displacement for
    the hip. Details about data files from Axis Neuron?
    2016-11-07

    Params:
    -------
    fname (str)
        Name of file to load
    includeDisplacement (bool=False)
        If displacement data is included in given bvh file for everything including root.
    removeBlank (bool=True)
        Remove entries where nothing changes over the entire recording session. This should mean that there
        was nothing being recorded in that field.

    Value:
    ------
    df (dataFrame)
    dt (float)
        Frame rate. Columns xx, yy, zz are positions and y, x, z are the Euler angles (assuming that the
        rotation matrices were output in YXZ order as is defulat in Axis Neuron).
    skeleton (list)
    """
    from itertools import chain
    from pyparsing import nestedExpr
    import string

    skeleton = load_skeleton(fname)
    bodyParts = skeleton.nodes

    # Find the line where data starts.
    lineix = 0
    with open(fname) as f:
        f.readline()
        f.readline()
        ln = f.readline()
        lineix += 3
        while not 'MOTION' in ln:
            ln = f.readline()
            lineix += 1
        
        # Read in the frame rate.
        while 'Frame Time' not in ln:
            ln = f.readline()
            lineix += 1
        dt = float( ln.split(' ')[-1] )
    
    # Parse motion.
    df = pd.read_csv(fname,skiprows=lineix+2,delimiter=' ',header=None)
    df = df.iloc[:,:-1]  # remove bad last col
    
    if includeDisplacement:
        df.columns = pd.MultiIndex.from_arrays([list(chain.from_iterable([[b]*6 for b in bodyParts])),
                                            ['xx','yy','zz','y','x','z']*len(bodyParts)])
    else:
        df.columns = pd.MultiIndex.from_arrays([[bodyParts[0]]*6 + 
                                                 list(chain.from_iterable([[b]*3 for b in bodyParts[1:]])),
                                            ['xx','yy','zz']+['y','x','z']*len(bodyParts)])
   

    # Filtering.
    if removeBlank:
        # Only keep entries that change at all.
        df = df.iloc[:,np.diff(df,axis=0).sum(0)!=0] 
    # Units of radians and not degress.
    df *= np.pi/180.
    return df,dt,skeleton

def load_skeleton(fname):
    """
    Load skeleton from BVH file header. 
    2016-11-07

    Params:
    -------
    fname (str)
        Name of file to load
    includeDisplacement (bool=False)
        If displacement data is included for everything including root.
    removeBlank (bool=True)
        Remove entries where nothing changes over the entire recording session. This should mean that there
        was nothing being recorded in that field.

    Value:
    ------
    df (dataFrame)
    dt (float)
        Frame rate.
    """
    from itertools import chain
    from pyparsing import nestedExpr
    import string,os

    # Parse skeleton.
    # Find the line where data starts and get skeleton tree lines.
    s = ''
    lineix = 0
    bodyParts = ['Hips']  # for keeping track of order of body parts
    with open(fname,'r') as f:
        f.readline()
        f.readline()
        ln = f.readline()
        lineix += 3
        while not 'MOTION' in ln:
            if 'JOINT' in ln:
                bodyParts.append( ''.join(a for a in ln.lstrip(' ').split(' ')[1] if a.isalnum()) )
            s += ln
            ln = f.readline()
            lineix += 1
        
        # Read in the frame rate.
        while 'Frame Time' not in ln:
            ln = f.readline()
            lineix += 1
        dt = float( ln.split(' ')[-1] )
    
    s = nestedExpr('{','}').parseString(s).asList()
    nodes = []

    def parse(parent,thisNode,skeleton):
        """
        Keep track of traversed nodes in nodes list.

        Params:
        -------
        parent (str)
        skeleton (list)
            As returned by pyparsing
        """
        children = []
        for i,ln in enumerate(skeleton):
            if (not type(ln) is list) and 'JOINT' in ln:
                children.append(skeleton[i+1])
            elif type(ln) is list:
                if len(children)>0:
                    parse(thisNode,children[-1],ln)
        nodes.append( Node(thisNode,parents=[parent],children=children) )
    
    # Parse skeleton.
    parse('','Hips',s[0])
    # Resort into order of motion data.
    nodesNames = [n.name for n in nodes]
    bodyPartsIx = [nodesNames.index(n) for n in bodyParts]
    nodes = [nodes[i] for i in bodyPartsIx]
    skeleton = Tree(nodes) 
   
    return skeleton

def subtract_hips_from_bvh(fname,dr,replace_file=True,temp_file='temp_bvh.bvh'):
    """
    Set hip displacement to 0 in given BVH file.

    Params:
    -------
    fname (str)
        Name of file to load
    dr (str)
    replace_file (bool=True)
    temp_file (str="temp_bvh.bvh")
    """
    import os
    # Check if file extension is.
    splitfname = fname.split('.')
    if len(splitfname)==1 or splitfname[1]!='bvh':
        fname += '.bvh'
    tempf = open('%s/%s'%(dr,temp_file),'w')
    
    skeleton = load_skeleton('%s/%s'%(dr,fname))
    bodyParts = skeleton.nodes
    
    # Find the line where data starts.
    with open('%s/%s'%(dr,fname)) as f:
        ln = f.readline()
        while 'Frame Time' not in ln:
            tempf.write(ln)
            ln = f.readline()
        tempf.write(ln)
        
        # Parse motion.
        for ln in f:
            ln = ln.split(' ')
            ln[:3] = ['0.000000','0.000000','0.000000']
            tempf.write(' '.join(ln))
    tempf.close()
    
    if replace_file:
        os.rename('%s/%s'%(dr,temp_file),'%s/%s'%(dr,fname))

def _parse_hmd_line(s):
    """
    Parse a single line from the HMD output from UE4 and return a date string with three floats representing
    the three axes that were measured in that line.
    """
    from datetime import datetime
    s = s.split(' ')
    date = datetime.strptime(s[0],'%Y-%m-%dT%H:%M:%S.%f')
    xyz = [float(i.split('=')[1]) for i in s[1:]]
    return date,xyz

def read_hmd_orientation_position(fname,dr=''):
    """
    Read in HMD rotations and position as output from OR blueprint.

    Params:
    -------
    fname (str)
        Path to file.

    Returns:
    --------
    rotationT (ndarray)
    rotation
    positionT
    position
    """
    if len(dr)==0:
        fname = '%s/%s'%(dr,fname)
    from datetime import datetime
    
    rotationT,positionT = [],[]
    rotation,position = [],[]
    with open(fname,'r') as f:
        f.readline()
        ln = f.readline()
        while not 'Position' in ln:
            d,r = _parse_hmd_line(ln)
            rotationT.append(d)
            rotation.append(r)
            ln = f.readline()

        for ln in f:
            d,p = _parse_hmd_line(ln)
            positionT.append(d)
            position.append(p)
    
    rotationT = np.array(rotationT)
    positionT = np.array(positionT)
    rotation = np.vstack(rotation)
    position = np.vstack(position)
    return rotationT,rotation,positionT,position

def load_hmd(fname,dr='',t=None,time_as_dt=True):
    """
    Read in data from HMD file and interpolate it to be in the desired uniform time units with given time
    points.

    Params:
    -------
    fname (str)
    dr (str='')
    t (ndarray)
        Time points to evaluate at. Assuming that this start at 0.
    interp_kwargs (dict={'kind':'linear'})

    Returns:
    --------
    Depending on input. If t is None:
    rotationT,rotation,positionT,position

    If t is given:
    t (ndarray)
        Only times that we have HMD data on.
    hmdrotX
    hmdposX
    hmdrotV
    hmdposV
    """
    if len(dr)>0:
        fname = '%s/%s'%(dr,fname)

    rotT,rot,posT,pos = read_hmd_orientation_position(fname)

    # Convert time stamps into seconds.
    if time_as_dt:
        rotT -= rotT[0]
        posT -= posT[0]
        rotT = np.array([i.total_seconds() for i in rotT])
        posT = np.array([i.total_seconds() for i in posT])

    if t is None:
        return rotT,rot,posT,pos
    
    interprot = interp1d(rotT,rot,kind='linear',axis=0,bounds_error=False,fill_value=np.nan)(t)
    interppos = interp1d(posT,pos,kind='linear',axis=0,bounds_error=False,fill_value=np.nan)(t)

   # Truncate at end of data.
    nanix = t>rotT[-1]
    t = t[nanix==0]
    interprot = interprot[nanix==0]
    interppos = interppos[nanix==0]

    # Use Savitzky-Golay filter with same default settings as with Vicon. 
    hmdrotV = savgol_filter(interprot,61,3,deriv=1,delta=t[1]-t[0],axis=0)
    hmdposV = savgol_filter(interppos,61,3,deriv=1,delta=t[1]-t[0],axis=0)
    
    return t,interprot,interppos,hmdrotV,hmdposV

def load_csv(fname,dr=''):
    """
    Load csv exported from Mokka. First column is time.

    Params:
    -------
    fname (str)
    dr (str='')
    """
    if len(dr)>0:
        fname = '%s/%s'%(dr,fname)
    
    df = pd.read_csv(fname,skiprows=5)
    df.drop([0,1],inplace=True)
    df.columns = pd.MultiIndex.from_product([df.columns[1::3].tolist(),['x','y','z']]).insert(0,'Time')
    return df

def load_visibility(fname,dr=''):
    """
    Load visible/invisible toggle times.

    Params:
    -------
    fname (str)
    dr (str='')

    Returns:
    --------
    visible (ndarray)
        First time is when animation starts.
    invisible (ndarray)
    """
    from datetime import datetime

    if len(dr)>0:
        fname = '%s/%s'%(dr,fname)
        
    visible,invisible = [],[]
    with open(fname,'r') as f:
        f.readline()
        ln = f.readline().rstrip()
        while not 'Invisible' in ln:
            visible.append(datetime.strptime(ln,'%Y-%m-%dT%H:%M:%S.%f'))
            ln = f.readline().rstrip()

        for ln in f:
            invisible.append(datetime.strptime(ln.rstrip(),'%Y-%m-%dT%H:%M:%S.%f'))
    
    visible = np.array(visible,dtype=datetime)
    invisible = np.array(invisible,dtype=datetime)
    return visible,invisible

def window_specs(person,dr):
    """
    Get when the different visible/invisible cycles occur in the given experiment. These data are obtained
    from visibility text files output from UE4.
    
    Returns:
    --------
    windowsByPart (dict)
        Keys correspond to trial types. Each dict entry is a list of tuples 
        ((type of window),(window start, window end))
    """
    from workspace.utils import load_pickle

    # Load AN subject data.
    load_pickle('%s/%s'%(dr,'quickload_an_port_vr.p'))
    windowsByPart = {}
    
    for trialno,part in enumerate(['avatar','avatar0','hand','hand0']):
        if part.isalpha():
            fname = part+'_visibility.txt'
        else:
            fname = part[:-1]+'_visibility_0.txt'

        visible,invisible = load_visibility(fname,dr)

        # Array denoting visible (with 1) and invisible (with 0) times.
        start = np.zeros((len(visible)+len(invisible)),dtype=object)
        start[::2] = visible
        start[1::2] = invisible
        start = np.array(map(lambda t:t.total_seconds(),np.diff(start)))
        start = np.cumsum(start)
        invisibleStart = start[::2]
        visibleStart = start[1::2]

        # Get the duration of the invisible and visible windows in the time series.
        mxLen = min([len(visibleStart),len(invisibleStart)])
        invDur = np.around(visibleStart[:mxLen]-invisibleStart[:mxLen],1)
        visDur = np.around(invisibleStart[1:][:mxLen-1]-visibleStart[:-1][:mxLen-1],1)
        windowDur = invDur[:-1]+visDur  # total duration cycle of visible and invisible

        # Identify the different types of windows that we have.
        windowSpecs = []
        windowIx = []
        for ix,i,w in zip(range(len(windowDur)),invDur[:-1],windowDur):
            if not (i,w) in windowSpecs:
                windowSpecs.append((i,w))
                windowIx.append([])
            windowIx[-1].append(ix)

        # Ignore all windows that appear when the flashing duration is changing. These
        # are identifiable as single appearances of a different visible cycle.
        ix = 0
        while ix<len(windowSpecs):
            if len(windowIx[ix])<3:
                windowIx.pop(ix)
                windowSpecs.pop(ix)
            else:
                ix += 1

        # Check that all indices are consecutive.
        for ix in windowIx:
            assert all(np.diff(ix)==1), ix

        # Identify the times at which these cycles start and end.
        windowStart,windowEnd = [],[]
        for ix in range(len(windowSpecs)):
            windowStart.append(invisible[windowIx[ix][0]])
            windowEnd.append(visible[windowIx[ix][-1]])

        windowsByPart[part] = zip(windowSpecs,zip(windowStart,windowEnd))

        # In buggy trials where animation did not stop properly, there is a segment at the end that
        # needs to be removed, corresponding to when the avatar was flashing very quickly.
        if windowsByPart[part][-1][0][0]==0:
            windowsByPart[part] = windowsByPart[part][:-1]
    return windowsByPart



# ------------------ #
# Class definitions. #
# ------------------ #
class VRTrial(object):
    def __init__(self,person,modelhandedness,rotation,dr):
        """
        Params:
        -------
        person (str)
        modelhandedness (list of str)
        rotation (list of float)
        dr (str)

        Attributes:
        -----------
        person
        modelhandedness
        rotation
        dr
        subjectTrial (dict)
            Full Axis Neuron trial data labeled by part+'T' part+'V'.
        templateTrial (dict)
            Full MotionBuilder trial data labeled by part+'T' part+'V'.
        timeSplitTrials
        subjectSplitTrials
        templateSplitTrials

        Methods:
        --------
        info
        subject_by_window_dur
        subject_by_window_spec
        pickle_trial_dicts
        pickle_phase
        """
        self.person = person
        self.modelhandedness = modelhandedness
        self.rotation = rotation
        self.dr = dr
        
        try:
            data = pickle.load(open('%s/trial_dictionaries.p'%self.dr,'rb'))
        except Exception:
            self.pickle_trial_dicts()
            
        data = pickle.load(open('%s/trial_dictionaries.p'%self.dr,'rb'))
        self.templateTrial = data['templateTrial']
        self.subjectTrial = data['subjectTrial']
        self.timeSplitTrials = data['timeSplitTrials']
        self.templateSplitTrials = data['templateSplitTrials']
        self.subjectSplitTrials = data['subjectSplitTrials']
        self.windowsByPart = data['windowsByPart']

    def info(self):
        print "Person %s"%self.person
        print "Trials available:"
        for part in ['avatar','avatar0','hand','hand0']:
            print "%s\tInvisible\tTotal"%part
            for spec,_ in self.windowsByPart[part]:
                print "\t%1.2f\t\t%1.2f"%(spec[0],spec[1])
    
    def subject_by_window_dur(self,windowDur,part):
        """
        Params:
        -------
        windowDur (list)
            Duration of visible/invisible cycle.
        part (str)
            Body part to return.
            
        Returns:
        --------
        selection (list)
            List of trials that have given window duration. Each tuple in list is a tuple of the 
            ( (invisible,total window), time, extracted velocity data ).
        """
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        if len(ix)==0:
            return
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection
    
    def template_by_window_dur(self,windowDur,part):
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        if len(ix)==0:
            return
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        return selection
    
    def subject_by_invisible_dur(self,windowDur,part):
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        if len(ix)==0:
            return
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.subjectSplitTrials[part][i] ))
        return selection

    def subject_by_window_spec(self,windowSpec,trialType):
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if spec in windowSpec:
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[trialType][i][0],
                               self.timeSplitTrials[trialType][i],
                               self.subjectSplitTrials[trialType][i] ))
        
        if trialType.isalpha():
            return selection + self.subject_by_window_spec(windowSpec,trialType+'0')
        return selection

    def template_by_window_spec(self,windowSpec,trialType):
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if spec in windowSpec:
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[trialType][i][0],
                               self.timeSplitTrials[trialType][i],
                               self.templateSplitTrials[trialType][i] ))
        if trialType.isalpha():
            return selection + self.template_by_window_spec(windowSpec,trialType+'0')
        return selection

    def template_by_invisible_dur(self,windowDur,part):
        ix = []
        i=0
        for spec,_ in self.windowsByPart[part]:
            if np.isclose(windowDur,spec[0]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[part][i][0],
                               self.timeSplitTrials[part][i],
                               self.templateSplitTrials[part][i] ))
        if trialType.isalpha():
            return selection + self.template_by_invisible_dur(windowSpec,trialType+'0')
        return selection

    def visibility_by_window_spec(self,windowSpec,trialType):
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if spec in windowSpec:
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            selection.append(( self.windowsByPart[trialType][i][0],
                               self.timeSplitTrials[trialType][i],
                               self.templateSplitTrials[trialType+'visibility'][i] ))
        if trialType.isalpha():
            return selection + self.visibility_by_window_spec(windowSpec,trialType+'0')
        return selection

    def phase_by_window_dur(self,source,windowDur,trialType):
        """
        Return instantaneous phase from bandpass filtered velocities on trial specificied by window
        duration.

        Params:
        -------
        source (str)
        windowDur (list of floats)
        trialType (str)
            'avatar', 'avatar0', 'hand', 'hand0'
        """
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if np.isclose(windowDur,spec[1]):
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            try:
                if source=='subject' or source=='s':
                    phases = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))['phases']
                elif source=='template' or source=='t':
                    phases = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))['phases']

                phases = [np.vstack(p) for p in phases]
                selection.append(( self.windowsByPart[trialType][i][0],phases ))
            except IOError:
                print "Trial %d in trial type %s not found."%(i,trialType)
        return selection

    def phase_by_window_spec(self,source,windowSpec,trialType):
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if spec in windowSpec:
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            try:
                if source=='subject' or source=='s':
                    data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                    phases,vs = data['phases'],data['vs']
                elif source=='template' or source=='t':
                    data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                    phases,vs = data['phases'],data['vs']
                
                phases = [np.vstack(p) for p in phases]
                selection.append(( self.windowsByPart[trialType][i][0],phases ))
            except IOError:
                print "Trial %d in trial type %s not found."%(i,trialType)
        return selection

    def filtv_by_window_spec(self,source,windowSpec,trialType,search_all=True):
        """
        Returns:
        --------
        list of twoples (windowSpec, filtv) where filtv is a list of 3 arrays corresponding to each dimension
        """
        ix = []
        i = 0
        for spec,_ in self.windowsByPart[trialType]:
            if spec in windowSpec:
                ix.append(i)
            i += 1
        
        selection = []
        for i in ix:
            if source=='subject' or source=='s':
                data = pickle.load(open('%s/subject_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            elif source=='template' or source=='t':
                data = pickle.load(open('%s/template_phase_%s_%d.p'%(self.dr,trialType,i),'rb'))
                phases,vs = data['phases'],data['vs']
            else:
                raise Exception

            vs = [np.vstack(p) for p in vs]
            selection.append(( self.windowsByPart[trialType][i][0],vs ))

        if trialType.isalpha() and search_all:
            return selection + self.filtv_by_window_spec(source,windowSpec,trialType+'0',False)

        return selection

    def dphase_by_window_dur(self,windowDur,trialType):
        """
        Difference in phase between subject and template motion.
        """
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_dur('s',windowDur,trialType)
        templatePhase = self.phase_by_window_dur('t',windowDur,trialType)
        dphase = []
        
        for i in xrange(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        if trialType.isalpha():
            return dphase + self.dphase_by_window_dur(windowDur,trialType+'0')
        return dphase

    def dphase_by_window_spec(self,windowSpec,trialType):
        """
        Difference in phase between subject and template motion.
        """
        from misc.angle import mod_angle
        
        subjectPhase = self.phase_by_window_spec('s',windowSpec,trialType)
        templatePhase = self.phase_by_window_spec('t',windowSpec,trialType)
        dphase = []
            
        for i in xrange(len(subjectPhase)):
            dphase.append(( subjectPhase[i][0], 
                            [mod_angle( s-t ) for s,t in zip(subjectPhase[i][1],templatePhase[i][1])] ))
        if trialType.isalpha():
            return dphase + self.dphase_by_window_spec(windowSpec,trialType+'0')
        return dphase

    def pickle_trial_dicts(self):
        """
        Put data for analysis into easily accessible pickles. Right now, I have visibility and hand 
        velocities for AN port data and motionbuilder files.
        """
        from pipeline import extract_motionbuilder_model2,extract_AN_port

        # Load AN data.
        df = pickle.load(open('%s/%s'%(self.dr,'quickload_an_port_vr.p'),'rb'))['df']
        windowsByPart = window_specs(self.person,self.dr)
        
        # Sort trials into the hand, arm, and avatar trial dictionaries: subjectTrial,
        # templateTrial, hmdTrials.
        subjectTrial,templateTrial,hmdTrials = {},{},{}
        timeSplitTrials,subjectSplitTrials,templateSplitTrials = {},{},{}

        for trialno,part in enumerate(['avatar','avatar0','hand','hand0']):
            print "Processing %s..."%part
            # Select time interval during which the trial happened.
            if part.isalpha():
                visible,invisible = load_visibility(part+'_visibility.txt',self.dr)
            else:
                visible,invisible = load_visibility(part[:-1]+'_visibility_0.txt',self.dr)
            startEnd = [visible[0],visible[-1]]

            mbT,mbV = extract_motionbuilder_model2(part,startEnd[0],self.modelhandedness[trialno])
            showIx = (mbT>startEnd[0]) & (mbT<startEnd[1])
            templateTrial[part+'T'],templateTrial[part+'V'] = mbT[showIx],mbV[showIx]

            anT,anX,anV,anA = extract_AN_port(df,self.modelhandedness[trialno],
                                              rotation_angle=self.rotation[trialno])
            showIx = (anT>startEnd[0]) & (anT<startEnd[1])
            subjectTrial[part+'T'],subjectTrial[part+'V'] = anT[showIx],anV[0][showIx]

            # Put trajectories on the same time samples so we can pipeline our regular computation.
            # Since the AN trial starts after the mbTrial...the offset is positive.
            subjectTrial[part+'V'],subjectTrial[part+'T'] = match_time(subjectTrial[part+'V'],
                               subjectTrial[part+'T'],
                               1/60,
                               offset=(subjectTrial[part+'T'][0]-templateTrial[part+'T'][0]).total_seconds(),
                               use_univariate=True)
            templateTrial[part+'V'],templateTrial[part+'T'] = match_time(templateTrial[part+'V'],
                                                                         templateTrial[part+'T'],
                                                                         1/60,
                                                                         use_univariate=True)
            

            # Times for when visible/invisible windows start.
            start = np.zeros((len(visible)+len(invisible)),dtype=object)
            start[::2] = visible
            start[1::2] = invisible
            start = np.array(map(lambda t:t.total_seconds(),np.diff(start)))
            start = np.cumsum(start)
            invisibleStart = start[::2]
            visibleStart = start[1::2] 

            visibility = np.ones_like(templateTrial[part+'T'])
            for i,j in zip(invisibleStart,visibleStart):
                visibility[(templateTrial[part+'T']>=i) & (templateTrial[part+'T']<j)] = 0
            if len(visible)<len(invisible):
                visibility[(templateTrial[part+'T']>=invisible[-1])] = 0
            templateTrial[part+'visibility'] = visibility

            timeSplitTrials[part],subjectSplitTrials[part],templateSplitTrials[part] = [],[],[]
            templateSplitTrials[part+'visibility'] = []
            for spec,startendt in windowsByPart[part]:
                startendt = ((startendt[0]-startEnd[0]).total_seconds(),
                             (startendt[1]-startEnd[0]).total_seconds())
                timeix = (subjectTrial[part+'T']<=startendt[1])&(subjectTrial[part+'T']>=startendt[0])
                t = subjectTrial[part+'T'][timeix]
                
                timeSplitTrials[part].append(t)
                subjectSplitTrials[part].append( subjectTrial[part+'V'](t) )
                templateSplitTrials[part].append( templateTrial[part+'V'](t) )

                timeix = (templateTrial[part+'T']<=startendt[1])&(templateTrial[part+'T']>=startendt[0])
                t = templateTrial[part+'T'][timeix]
                templateSplitTrials[part+'visibility'].append( visibility[timeix] )
            
            # Get the beginning fully visible window. Inser this into the beginning of the list.
            windowsByPart[part].insert(0,((0,0),(0,invisibleStart[0])))
            timeix = (subjectTrial[part+'T']>=0)&(subjectTrial[part+'T']<=invisibleStart[0])
            t = subjectTrial[part+'T'][timeix]
            
            timeSplitTrials[part].insert(0,t)
            subjectSplitTrials[part].insert( 0,subjectTrial[part+'V'](t) )
            templateSplitTrials[part].insert( 0,templateTrial[part+'V'](t) )

            timeix = (templateTrial[part+'T']<=invisibleStart[0])&(templateTrial[part+'T']>=0)
            templateSplitTrials[part+'visibility'].insert( 0,visibility[timeix] )

        
        pickle.dump({'templateTrial':templateTrial,
                     'subjectTrial':subjectTrial,
                     'timeSplitTrials':timeSplitTrials,
                     'templateSplitTrials':templateSplitTrials,
                     'subjectSplitTrials':subjectSplitTrials,
                     'windowsByPart':windowsByPart},
                    open('%s/trial_dictionaries.p'%self.dr,'wb'),-1)

    def pickle_phase(self,trial_types=['avatar','avatar0','hand','hand0']):
        """
        Calculate bandpass filtered phase and pickle.
        """
        from pipeline import pipeline_phase_calc
        
        for part in trial_types:
            nTrials = len(self.windowsByPart[part])  # number of trials for that part

            # Subject.
            toProcess = []
            trialNumbers = []
            for i in xrange(nTrials):
                # Only run process if we have data points. Some trials are missing data points.
                # NOTE: At some point the min length should made to correspond to the min window
                # size in the windowing function for filtering.
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.subjectSplitTrials[part][i][:,0],
                                        self.subjectSplitTrials[part][i][:,1],
                                        self.subjectSplitTrials[part][i][:,2])) )
                else:
                    print "Ignoring %s trial no %d with windowspec (%1.1f,%1.1f)."%(part,i,
                        self.windowsByPart[part][i][0][0],self.windowsByPart[part][i][0][1])
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['subject_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])
            # Template.
            toProcess = []
            trialNumbers = []
            for i in xrange(nTrials):
                if len(self.timeSplitTrials[part][i])>501:
                    trialNumbers.append(i)
                    toProcess.append( (self.timeSplitTrials[part][i],
                                       (self.templateSplitTrials[part][i][:,0],
                                        self.templateSplitTrials[part][i][:,1],
                                        self.templateSplitTrials[part][i][:,2])) )
            pipeline_phase_calc(trajs=toProcess,dr=self.dr,
                                file_names=['template_phase_%s_%d'%(part,i)
                                            for i in trialNumbers])
# end VRTrial


class Node(object):
    def __init__(self,name=None,parents=[],children=[]):
        self.name = name
        self.parents = parents
        self.children = children

    def add_child(self,child):
        self.children.append(child)

class Tree(object):
    def __init__(self,nodes):
        """
        Data structure for BVH skeleton hierarchy.

        Attributes:
        -----------
        _nodes (Node)
        nodes
        adjacency
        """
        self._nodes = nodes
        self.nodes = [n.name for n in nodes]
        names = [n.name for n in nodes]
        if len(np.unique(names))<len(names):
            raise Exception("Nodes have duplicate names.")

        self.adjacency = np.zeros((len(nodes),len(nodes)))
        for i,n in enumerate(nodes):
            for c in n.children:
                try:
                    self.adjacency[i,names.index(c)] = 1
                # automatically insert missing nodes (these should all be dangling)
                except ValueError:
                    self.adjacency = np.pad( self.adjacency, ((0,1),(0,1)), mode='constant', constant_values=0)
                    self._nodes.append( Node(c) )
                    names.append(c)

                    self.adjacency[i,names.index(c)] = 1
        
    def print_tree(self):
        print self.adjacency
    
    def parents(self,node):
        """
        Return parents of particular node.

        Returns:
        --------
        parents (list)
            Parents starting from immediate parent and ascending up the tree.
        """
        parents = []
        iloc = self.nodes.index(node)

        while np.any(self.adjacency[:,iloc]):
            iloc = np.where(self.adjacency[:,iloc])[0][0]
            parents.append(self.nodes[iloc])

        return parents
