# ================================================================================================ # 
# Reading and cleaning data from UE4.
# ================================================================================================ # 

from __future__ import division
from utils import *

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

def load_visibility(fname,dr=''):
    """
    Load visible/invisible toggle times.

    Parameters
    ----------
    fname : str
    dr : str,''

    Returns
    -------
    visible : ndarray of datetime.datetime
        First time is when animation starts.
    invisible : ndarray of datetime.datetime
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
    Get when the different visible/invisible cycles occur in the given experiment. These data are
    obtained from visibility text files output from UE4.
    
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


