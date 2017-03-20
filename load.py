# Module for loading and extracting data from Axis Neuron files.
#
# Classes:
# Node, Tree
# 
# Author: Edward D. Lee
# Email: edl56@cornell.edu
# 2016-08-11

from __future__ import division
import pandas as pd
import numpy as np
from utils import *

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
          ('Caeli (J) Eddie (J) Hands Low Light','20170317')]

def get_dr(fname,date=None):
    """Return directory where files are saved."""
    from os.path import expanduser
    homedr = expanduser('~')
    datadr = 'Dropbox/Documents/Noitom/Axis Neuron/Motion Files'

    if not date is None:
        return {'20170307':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie_Occlusion'),
                '20170310':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie_Startup'),
                '20170317':'%s/%s/%s_%s'%(homedr,datadr,date,'Caeli_Eddie')}[date]

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

def filter_hand_trials(filesToFilter,dt=1/60,
        extract_calc_kwargs={'rotate_to_face':True,
                             'remove_hip_drift':True,
                             'dotruncate':5},
        filterparams='default'):
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
    from utils import smooth
    import cPickle as pickle
    
    bodyparts = [['RightHand','LeftHand'],
                 ['LeftHand','RightHand']]

    for fileix in filesToFilter:
        # Read position, velocity and acceleration data from files.
        if type(get_fnames()[fileix]) is tuple:
            fname,date = get_fnames()[fileix]
            T,leaderX,leaderV,leaderA,followerX,followerV,followerA = extract_calc(fname,
                                                                                   get_dr(fname,date),
                                                                                   bodyparts,
                                                                                   dt,**extract_calc_kwargs)
        else:
            fname = get_fnames()[fileix]
            T,leaderX,leaderV,leaderA,followerX,followerV,followerA = extract_calc(fname,
                                                                                   get_dr(fname),
                                                                                   bodyparts,
                                                                                   dt,**extract_calc_kwargs)

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
        
        try:
            pickle.dump({'T':T,
                         'leaderX':leaderX,'followerX':followerX,
                         'leaderV':leaderV,'followerV':followerV,
                         'leaderA':leaderA,'followerA':followerA},
                        open('%s/%s.p'%(get_dr(fname,date),fname),'wb'),-1)
        except NameError:
            pickle.dump({'T':T,
                         'leaderX':leaderX,'followerX':followerX,
                         'leaderV':leaderV,'followerV':followerV,
                         'leaderA':leaderA,'followerA':followerA},
                        open('%s/%s.p'%(get_dr(fname),fname),'wb'),-1)


def load_calc(fname,cols='V',read_csv_kwargs={}):
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

    df = pd.read_csv(fname,skiprows=5,sep='\t',**read_csv_kwargs)
    
    # Only keep desired columns.
    keepix = np.zeros((len(df.columns)),dtype=bool)
    for s in cols:
        keepix += np.array([s in c for c in df.columns])
    df = df.ix[:,keepix]
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
    with open(fname,'r') as f:
        zd = np.array([float(i) for i in f.readline().split('\t')[1:]])
    #n = np.cross(zd,np.array([-1,0,0]))
    #theta = np.arccos(zd.dot([-1,0,0]))
    #for i in xrange(len(df.columns)):
    #    if any([c+'-x' in df.columns[i] for c in cols]):
    #        df.ix[:,i:i+3].values[:,:] = rotate(df.ix[:,i:i+3].values,n,theta)
    return df,zd

def extract_parts(df,bodyparts):
    """
    Extract columns for specific body parts from loaded calc file by comparing with column headers.
    2017-03-19
    """
    returnix = []
    columnBodyParts = [c.split('-')[0].lower() for c in df.columns]
    for i,c in enumerate(columnBodyParts):
        if c in bodyparts:
            returnix.append(i)
    return df.ix[:,returnix]

def group_cols(columns):
    """
    Group columns of 3 into multiindex with xyz subcolumns.
    2017-03-03
    """
    bodyparts = [c.split('-')[0] for c in columns[::3]]
    return pd.MultiIndex.from_product((bodyparts,['x','y','z'])) 

def extract_calc(fname,dr,bodyparts,dt,
                 append=True,
                 dotruncate=5,
                 remove_hip_drift=True,
                 rotate_to_face=True,
                 usezd=False,
                 read_csv_kwargs={},
                 center_x=False,
                ):
    """
    Extract specific set of body parts from calculation file. If a file with coordination of hands is given,
    then I have to align the subjects to a global coordinate frame defined by the initial orientation of their
    hands.

    For import of hands trials, the first axis is the direction along which the subjects are aligned.

    The slowest part is loading the data from file.
    2017-01-16

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
        for i,ix in enumerate(bodypartix[leaderix]):
            leaderX.append( leaderdf.values[:,ix*9:ix*9+3].copy() ) 
            leaderV.append( leaderdf.values[:,ix*9+3:ix*9+6].copy() ) 
            leaderA.append( leaderdf.values[:,ix*9+6:ix*9+9].copy() ) 
        for i,ix in enumerate(bodypartix[1-leaderix]):
            followerX.append( followerdf.values[:,ix*9:ix*9+3].copy() )
            followerV.append( followerdf.values[:,ix*9+3:ix*9+6].copy() )
            followerA.append( followerdf.values[:,ix*9+6:ix*9+9].copy() )
    else:
        for i,ix in enumerate(bodypartix[leaderix]):
            if i==0:
                leaderX = [leaderdf.values[:,ix*9:ix*9+3].copy()]
                leaderV = [leaderdf.values[:,ix*9+3:ix*9+6].copy()]
                leaderA = [leaderdf.values[:,ix*9+6:ix*9+9].copy()]
            else:
                leaderV[0] += leaderdf.values[:,ix*9+3:ix*9+6]
                leaderA[0] += leaderdf.values[:,ix*9+6:ix*9+9]
                
        for i,ix in enumerate(bodypartix[1-leaderix]):
            if i==0:
                followerX = [followerdf.values[:,ix*9:ix*9+3].copy()]
                followerV = [followerdf.values[:,ix*9+3:ix*9+6].copy()]
                followerA = [followerdf.values[:,ix*9+6:ix*9+9].copy()]
            else:
                followerV[0] += followerdf.values[:,ix*9+3:ix*9+6]
                followerA[0] += followerdf.values[:,ix*9+6:ix*9+9]

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
    for i,ix in enumerate(bodypartix[leaderix]):
        leaderW.append( leaderdf.values[:,ix*9:ix*9+3].copy() ) 
    for i,ix in enumerate(bodypartix[1-leaderix]):
        followerW.append( followerdf.values[:,ix*9:ix*9+3].copy() )
    
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
    Load data from BVD file. Euler angles are given as YXZ. Axis Neuron only keeps track of displacement for the hip.
    Details about data files from Axis Neuron?
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
    import string

    # Parse skeleton.
    # Find the line where data starts and get skeleton tree lines.
    s = ''
    lineix = 0
    bodyParts = ['Hips']  # for keeping track of order of body parts
    with open(fname) as f:
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



# ------------------ #
# Class definitions. #
# ------------------ #
class Node(object):
    def __init__(self,name=None,parents=[],children=[]):
        self.name = name
        self.parents = parents
        self.children = children

    def add_child(self,child):
        self.children.append(child)

class Tree(object):
    def __init__(self,nodes):
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

