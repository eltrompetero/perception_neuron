# ================================================================================================ # 
# Module for loading and extracting data from Axis Neuron calculation and UE4 data files.
#
# Author: Edward D. Lee
# Email: edl56@cornell.edu
# ================================================================================================ # 

from __future__ import division
from utils import *
import cPickle as pickle
import os

def extract_AN_port(df,modelhand,rotation_angle=0):
    """
    Take dataframe created from load_AN_port() and pull out the X, V, A data in the standard
    coordinate system.

    Parameters
    ----------
    df : pd.DataFrame
    modelhand : str
    rotation_angle : float,0
        Rotation of raw data about [0,0,1] local z-axis which is pointing into the ground.

    Returns
    -------
    T,X,V,A
    """
    from datetime import datetime

    anT = np.array(map(datetime.utcfromtimestamp,df['Timestamp'].values.astype(datetime)/1e9))
    
    # Extract only necessary body part from the dataframe.
    # extract_calc_solo handles reorientation of axes into standard coordinate system and rotation
    # about xy plane. This all happens in extract_calc_solo().
    df = load_calc('',cols='XVA',return_zd=False,df=df.iloc[:,1:])
    if modelhand=='Right':
        _,anX,anV,anA = extract_calc_solo(leaderdf=df,bodyparts=['LeftHand'],
                                          dotruncate=0,
                                          rotation_angle=rotation_angle,
                                          remove_hip_drift=False)
    else:
        _,anX,anV,anA = extract_calc_solo(leaderdf=df,bodyparts=['RightHand'],
                                          dotruncate=0,
                                          rotation_angle=rotation_angle,
                                          remove_hip_drift=False)

    return anT,anX,anV,anA

def left_hand_col_indices(add_one=True):
    """
    Get column indices that correspond to the velocities vx, vy,vz of the left hand. In the calc
    file, these cols start at 1 (and not 0) so these col indices are incremented by one. You must
    specify not to do this one if you want to use these as indices for a Python array.
    
    Parameters
    ----------
    add_one : bool,True

    Returns
    -------
    idx : list
        Indices of left hand velocities incremented by one if add_one is True.
    """
    columns = calc_file_headers()
    skeleton = calc_file_body_parts()
    nameIx = 0
    for i,s in enumerate(skeleton):
        if not 'contact' in s:
            for j,c in enumerate(columns):
                columns[j] = c.replace(str(nameIx+1).zfill(2),s)
            nameIx += 1
    if add_one:
        return [columns.index(p)+1 for p in ['LeftHand-V-x','LeftHand-V-y','LeftHand-V-z']]
    return [columns.index(p) for p in ['LeftHand-V-x','LeftHand-V-y','LeftHand-V-z']]

def right_hand_col_indices(add_one=True):
    """
    Get column indices that correspond to the velocities vx, vy,vz of the left hand. In the calc
    file, these cols start at 1 (and not 0) so these col indices are incremented by one. You must
    specify not to do this one if you want to use these as indices for a Python array.
    
    Parameters
    ----------
    add_one : bool,True

    Returns
    -------
    idx : list
        Indices of left hand velocities incremented by one if add_one is True.
    """
    columns = calc_file_headers()
    skeleton = calc_file_body_parts()
    nameIx = 0
    for i,s in enumerate(skeleton):
        if not 'contact' in s:
            for j,c in enumerate(columns):
                columns[j] = c.replace(str(nameIx+1).zfill(2),s)
            nameIx += 1
    if add_one:
        return [columns.index(p)+1 for p in ['RightHand-V-x','RightHand-V-y','RightHand-V-z']]
    return [columns.index(p) for p in ['RightHand-V-x','RightHand-V-y','RightHand-V-z']]

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

def load_calc(fname,cols='V',read_csv_kwargs={},return_zd=True,df=None):
    """
    Load calculation file output by Axis Neuron with cols renamed to be user friendly. 
    Note that z-axis points into the ground by default.
    2016-12-05

    Parameters
    ----------
    fname : str
    skeleton : list of str
        Names fo the bones specified in fname.
    cols : str
        Data columns to keep. Columns are XVQAW (position, vel, quaternion, acc, angular vel). Order
        does not matter.
    read_csv_kwargs : dict
        kwargs for pandas.read_csv
    return_zd : bool,True
        Return direction in which user was initially facing.
    df : pandas.DataFrame
        Supply an already loaded calc file to read from.

    Returns
    -------
    df : pandas.DataFrame
        Loaded calc file. Cols have been renamed so that numbers have been replaced by strings.
    zd : ndarray
        Vector of direction user was initially facing.

    Example
    -------
    >>> df,zd = load_calc('Eddie.calc',cols='V')
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
    if return_zd:
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

def extract_calc_solo(fname='',
                      dr='',
                      bodyparts=[],
                      dt=1/120,
                      leaderdf=None,
                      append=True,
                      dotruncate=0,
                      remove_hip_drift=True,
                      read_csv_kwargs={},
                      center_x=False,
                      orient_before_rotation=True,
                      rotation_angle=False
                      ):
    """
    Extract specific set of body parts from calculation file with one individual. This is modification of
    extract_calc(). 

    Parameters
    ----------
    fname (str)
    dr (str)
    bodyparts (list of strings)
        Body parts to keep.
    dt (float)
    leaderdf (pandas.DataFrame=None)
        If given, this will be the data array used to extract data.
    append (bool=True)
        If true, keep list of data from bodyparts else add all the velocities and acceleration
        together. This is useful if we're looking at the motion of the feet and want to look at the
        sum of the motion of the feet (because we don't care about stationary feet).
    dotruncate (float=5)
        Truncate beginning and end of data by this many seconds.
    remove_hip_drift (bool=True)
    read_csv_kwargs (dict)
        Passed onto pandas.read_csv
    center_x (bool=False)
        Subtract mean from the mean of each body parts' displacement.
    orient_before_rotation : bool,True
        Orient into standard coordinate system before rotating. This means negating the YX axes
        for PN data.
    rotation_angle (int=False)
        If an integer or float, X, V, A will be rotated about the local [0,0,1] z-axis. Note that this z-axis
        points into the ground.

    Returns
    -------
    T,X,V,A : list of DataFrames
    """
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

    if orient_before_rotation:
         for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:,1:] *= -1
            v[:,1:] *= -1
            a[:,1:] *= -1

    if rotation_angle:
        for x,v,a in zip(leaderX,leaderV,leaderA):
            x[:,:2] = rotate_xy(x[:,:2],rotation_angle)
            v[:,:2] = rotate_xy(v[:,:2],rotation_angle)
            a[:,:2] = rotate_xy(a[:,:2],rotation_angle)

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
