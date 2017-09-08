# ===================================================================================== #
# Module for reading data from Axis Neuron broadcast port.
# Author: Eddie Lee edl56@cornell.edu
# ===================================================================================== #
from __future__ import division
import numpy as np
from datetime import datetime,timedelta
import os,time,socket,shutil
import pandas as pd
from load import calc_file_headers

HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = 7003  # Calculation data.
DATADR = os.path.expanduser('~')+'/Dropbox/Sync_trials/Data'

# Functions for reading from broadcasting port.
def read_port():
    def incoming(host, port):
        """Open specified port and return file-like object"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.connect((host,port))
        # In worst case scenario, all 947 fields will be 9 bytes with 1 space in between and a \n
        # character.
        t,data = datetime.now(),sock.recv(9470*2)
        sock.close()
        return t,data
    
    data = ''
    while data=='':
        t,data = incoming(HOST, PORT)
    
    # Clean up data that is returned from port to be a single line.
    lastn = len(data)-data[::-1].index('\n')-1
    try:
        plastn = len(data)-4-data[:lastn-2][::-1].index('\n')
    except ValueError:
        plastn = 0

    return t,data[plastn:lastn]

def format_port_output(s):
    s=s.split(' ')[1:11]
    f=[float(i) for i in s]
    return f

def _format_port_output(s):
    """For BVH file."""
    iloc=s.find('Caeli')
    if iloc==-1:
        return ['NaN']*10
    s=s[iloc:].split(' ')[1:11]
    f=[float(i) for i in s]
    return f
# end port section

def record_AN_port(fname):
    """
    Start recording data from Axis Neuron port when presence of start.txt is detected and stop when
    end.txt is detected in DATADR.

    Parameters
    ----------
    fname : str
    """
    f = open(fname,'w')

    # Check that recording has started as given by presence of lock file.
    while not os.path.isfile('C:/Users/Eddie/Dropbox/Sync_trials/Data/start.txt'):
        time.sleep(1)
    
    # Write header line.
    headers = list(calc_file_headers())
    headers[-1] = ''.join(headers[-1].split())  # Remove space in last column header.
    f.write('Timestamp '+' '.join(headers)+'\n')
    
    # Capture port output.
    while not os.path.isfile('C:/Users/Eddie/Dropbox/Sync_trials/Data/end.txt'):
        portOut = read_port()
        t = portOut[0].isoformat()
        f.write('%s %s\n'%(t,portOut[1].strip()))
        f.flush()
    
    f.close()

def _fix_problem_dates(f,fname):
    """
    Insert missing datetime or missing microseconds at beginning of line. Put in 1900-01-01T00:00:00.000 if
    missing date completely.
    """
    import uuid
    tmpfile = str(uuid.uuid4())
    with open('/tmp/'+tmpfile,'w') as fout:
        # skip header lines
        for i in xrange(5):
            fout.write(f.readline())

        for ln in f:
            try:
                d = datetime.strptime(ln[:26], '%Y-%m-%dT%H:%M:%S.%f')
            except ValueError:
                if len(ln[:26].split()[0])==19:
                    # Some values seem to be cutoff because microseconds is precisely 0.
                    #print "Inserting microseconds."
                    ln = ln.split()
                    ln[0] += '.000000'
                    ln = ' '.join(ln)+'\n'
                else:
                    ln = '1900-01-01T00:00:00.000000 '+ln
                # Sometimes, a single port broadcost seems to overlap with another.
                if len(ln.split())>948:
                    ln = ' '.join(ln.split()[:948])+'\n'
            fout.write(ln) 
    shutil.move('/tmp/'+tmpfile,fname)

def load_AN_port(fname,dr='',time_as_dt=True,n_avatars=1,fix_file=True,read_csv_kwargs={}):
    """
    With data from a single individual at this moment.
    
    Params:
    -------
    fname (str)
    dr (str='')
    time_as_dt (bool=True)
    fix_file (bool=True)
        Parse input file and fix any problems with dates.
    """
    if len(dr)>0:
        fname = '%s/%s'%(dr,fname)
    
    if fix_file:
        # Insert missing times for the file to be read in.
        with open(fname,'r') as f:
            _fix_problem_dates(f,fname) 

    # Read in start and stop times at beginning of file.
    #with open(fname,'r') as f:
    #    startTime = datetime.strptime( f.readline().split(' ')[-1] )
    #    stopTime = datetime.strptime( f.readline().split(' ')[-1] )
            
    df = pd.read_csv(fname,delimiter=' ',skiprows=3,**read_csv_kwargs)
    df.iloc[:,0] = df.iloc[:,0].apply(lambda t: datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f'))

    # Linearly interpolate missing date times. Assuming that no two sequential data points are missing
    # times which seems to be the case...
    iloc = np.where( pd.DatetimeIndex(df['Timestamp']).year==1900 )[0]
    for i in iloc:
        if i>0 and i<(len(df)-1):
            df.iloc[i,0] = timedelta(seconds=(df.iloc[i+1,0]-df.iloc[i-1,0]).total_seconds()/2) + df.iloc[i-1,0]
    # Remove last data point if the time is uncertain.
    if pd.DatetimeIndex(df.tail(1)['Timestamp']).year==1900:
        df = df.iloc[:-1]

    if time_as_dt:
        # Convert time stamp into time differences in seconds. This means we have to remove the first data
        # point.
        dt = np.diff(df.iloc[:,0]).astype(int)/1e9
        df = df.iloc[1:,:]
        df['Timestamp'] = df['Timestamp'].apply(pd.to_numeric,errors='coerce')
        df['Timestamp'] = dt
    return df



# ---------------------------------------------------- #
# Functions for real time reading from port broadcast. #
# ---------------------------------------------------- #
def read_an_port_file(fopen,partsIx):
    """
    Read from AN port broadcast file and extract the information from specified fields.

    Parameters
    ----------
    fopen : file
        Read from this file at current place in file.
    partsIx : list of column indices to return

    Returns
    -------
    subT : ndarray
        subject time
    subV : ndarray
        subject velocity
    """
    subT = np.array(())
    subV = np.zeros((0,3))

    # Taking anything that has been added to the file from current read position.
    fopen.seek(0,1)
    ln = fopen.read() 
    # Take last line. 
    ln = ln.split('\n')[:-1]
    
    print "Accessing file..."
    for el in ln:
        # Some values seem to be cutoff because microseconds is precisely 0.
        el = el.split()
        if len(el[0])!=26:
            el[0] += '.000000'
        
        # Read in data.
        subT = np.append(subT,datetime.strptime(el[0], '%Y-%m-%dT%H:%M:%S.%f'))
        subV = np.append(subV,[[float(f) for f in [el[ix] for ix in partsIx]]],axis=0)
    assert len(subT)==len(subV)
    return subT,subV

def fetch_vel_history(fopen,partsIx,dt=3,return_datetime=False,t0=None):
    """
    Return the velocity history interpolated at a constant frame rate since the last query as specified by
    current read position in file. Be careful because this could mean parsing entire file if the
    read position is at beginning. Calls read_an_port_file().

    Parameters
    ----------
    fopen : file
        File where output from AN port broadcast is.
    partsIx : list of ints
        Indices of columns for which to extract data.
    dt : float,3
        Number of seconds to go into the past.
    t0 : datetime,None
        Time at which to start (should be before now). This is useful for aligning data with the
        right time spacing when reading from the file multiple times. The difference between this
        and the recording start time will be added to tdate.
    return_datetime : bool
        Return tdate.

    Returns
    -------
    v : ndarray
        Array of dimensions (n_time,3).
    t : ndarray
        Time in seconds relative to when function was called.
    tdate : ndarray
        Time as datetime.
    """
    from datetime import timedelta
    from utils import MultiUnivariateSpline

    # Extract data from file.
    subT,subV = read_an_port_file(fopen,partsIx)
    now = datetime.now()
    t = np.array([i.total_seconds() for i in subT-now])

    # Only keep time points within dt of last measurement.
    timeix = t>(t[-1]-dt)
    t = t[timeix]
    subV = subV[timeix]

    # MultiUnivariateSpline needs t that has been ordered from lowest to highest.
    splineV = MultiUnivariateSpline(t,subV)
    if not t0 is None:
        if t0<subT[0]:
            # If we exceed bounds, we will have to fill in the blanks after interpolation.
            interpIx = np.array([t0<i for i in subT])
            print "must interpolate %d points"%interpIx.sum()
        t = np.arange((t0-now).total_seconds(),t[-1],1/60)
        dt = (t0-subT[0]).total_seconds()
    else:
        t = np.arange(t[0],t[-1],1/60)
        dt = 0
        interpIx = np.zeros_like(subT)==1

    splineV = splineV(t)
    if interpIx.any():
        splineV[interpIx] = np.nan
    if return_datetime:
        tasdate = np.array([now+timedelta(0,i+dt) for i in t])
        return splineV,t,tasdate
    return splineV,t

def left_hand_col_indices():
    from load import calc_file_headers,calc_file_body_parts
    # Get the columns with the data that we're interested in.
    columns = calc_file_headers()
    skeleton = calc_file_body_parts()
    nameIx = 0
    for i,s in enumerate(skeleton):
        if not 'contact' in s:
            for j,c in enumerate(columns):
                columns[j] = c.replace(str(nameIx+1).zfill(2),s)
            nameIx += 1
    return [columns.index(p)+1 for p in ['LeftHand-V-x','LeftHand-V-y','LeftHand-V-z']]

def right_hand_col_indices():
    from load import calc_file_headers,calc_file_body_parts
    # Get the columns with the data that we're interested in.
    columns = calc_file_headers()
    skeleton = calc_file_body_parts()
    nameIx = 0
    for i,s in enumerate(skeleton):
        if not 'contact' in s:
            for j,c in enumerate(columns):
                columns[j] = c.replace(str(nameIx+1).zfill(2),s)
            nameIx += 1
    return [columns.index(p)+1 for p in ['RightHand-V-x','RightHand-V-y','RightHand-V-z']]

def fetch_matching_avatar_vel(avatar,part,t,disp=False):
    """
    Get the stretch of avatar velocities that aligns with the velocity data of the subject. Assumes
    that the start time for the avatar is given in ~/Dropbox/Sync_trials/Data/start.txt

    Parameters
    ----------
    avatar : dict
        This would be the templateTrial loaded in VRTrial.
    part : str
        Choose from 'avatar','avatar0','hand','hand0'.
    t : array of datetime objects
        Stretch of time to return data from.

    Returns
    -------
    v : ndarray
        Avatar's velocity that matches given time stamps.
    """
    with open('%s/%s'%(DATADR,'start.txt'),'r') as f:
        startt = datetime.strptime(f.readline(),'%Y-%m-%dT%H:%M:%S.%f')
    
    # Transform dt to time in seconds.
    t = np.array([i.total_seconds() for i in t-startt])
    if disp:
        print "Getting avatar times between %1.1fs and %1.1fs."%(t[0],t[-1])

    # Return part of avatar's trajectory that agrees with the stipulated time bounds.
    return avatar[part+'V'](t)



# ======= #
# Classes #
# ======= #
class HandSyncExperiment(object):
    def __init__(self,outfile,duration,trial_type,parts_ix):
        """
        Parameters
        ----------
        outfile : str
            Output file where to write current coherence value. This will be placed in DATADR.
        duration : float
            Seconds into past to analyze.
        trial_type : str
            'avatar','avatar0','hand','hand0'
        parts_ix : str
        """
        self.outfile = outfile
        self.duration = duration
        self.trialType = trial_type
        self.partsIx = parts_ix
    
    def start(self,update_delay=.25):
        """
        Start experiment. Will calculate coherence and output.
        
        NOTE: need to add error checking

        Parameters
        ----------
        update_delay : float
            Number of seconds to wait between updating arrays.
        """
        from load import subject_settings_v3,VRTrial
        from gpr import CoherenceEvaluator,GPR

        # Load avatar trajectory. This has a bunch of overhead because it shouldn't necessary to also load
        # a trial from a subject.
        subject_settings = subject_settings_v3
        person,modelhandedness,rotation,dr = subject_settings(0)
        trial = VRTrial(person,modelhandedness,rotation,dr)
        avatar = trial.templateTrial

        # Demo only: need a start time.
        fopen = open('%s/%s'%(DATADR,'start.txt'),'w')
        fopen.write(datetime.now().isoformat())
        fopen.close()


        # Run experiment.
        # Performance evaluation.
        ceval = CoherenceEvaluator(.035,10,60,90)
        
        # For retrieving the subject's velocities.
        subVBroadcast = ANBroadcast(self.duration,
                                    '%s/%s'%(DATADR,'an_port.txt'),
                                    self.partsIx)
        # Wait for data to be written to end of file..
        time.sleep(self.duration+1)

        # Get data from subject and also from avatar.
        with open('%s/%s'%(DATADR,self.outfile),'w') as fout:
            subVBroadcast.update()
            assert len(subVBroadcast.tdate)>100
            v = subVBroadcast.v
            avv = fetch_matching_avatar_vel(avatar,self.trialType,subVBroadcast.tdate,
                                            disp=True)

            while not os.path.isfile('%s/%s'%(DATADR,'end.txt')):
                avgcoh = ceval.evaluateCoherence(avv[:,2],v[:,2])
                fout.write('%f\n'%avgcoh)
                
                time.sleep(update_delay)
                
                subVBroadcast.update()
                v = subVBroadcast.v
                avv = fetch_matching_avatar_vel(avatar,self.trialType,subVBroadcast.tdate,
                                                disp=True)
# end HandSyncExperiment


class ANBroadcast(object):
    def __init__(self,duration,broadcast_file,parts_ix):
        """
        Class for keeping track of history of velocity.
        
        Parameters
        ----------
        duration : float
            Number of seconds to keep track in the past.
        broadcast_file : str
            Where data from AN port is broadcast.
        parts_ix : list of ints
            Indices of corresponding column headers in broadcast file.
        """
        self.duration = duration
        
        # Open file and go to the end.
        self.fin = open(broadcast_file)
        self.fin.seek(0,2)
        
        self.partsIx = parts_ix

        self.refresh()
    
    def refresh(self):
        """Clear stored arrays."""
        self.v = np.zeros((0,3))
        self.t = np.array(())
        self.tdate = np.array(())

    def update(self):
        """
        Update record of velocities by reading latest velocities and throwing out old velocities.
        
        Must interpolate missing points in between readings. Best way to do this cheaply is to use a
        Kalman filter to add a new point instead of refitting a global spline every time we get a
        new set of points.
        """
        vnew,tdatenew = self.fetch_new_vel()
        if len(vnew)==0:
            print "Nothing new to read."
            return
        
        # Update arrays with new data. 
        self.v = np.append(self.v,vnew,axis=0)
        self.tdate = np.append(self.tdate,tdatenew)
        now = datetime.now()
        self.t = np.array([(t-now).total_seconds() for t in self.tdate])
        
        # Only keep data points that are within self.duration of now.
        tix = self.t>-self.duration
        self.v = self.v[tix]
        self.t = self.t[tix]
        self.tdate = self.tdate[tix]
        
        self.interpolate()
    
    def fetch_new_vel(self):
        """
        Return the velocity history interpolated at a constant frame rate since the last query as specified by
        current read position in file. Be careful because this could mean parsing entire file if the
        read position is at beginning. Calls read_an_port_file().

        Returns
        -------
        v : ndarray
            Array of dimensions (n_time,3).
        tdate : ndarray
            Time as datetime.
        """
        from datetime import timedelta
        from utils import MultiUnivariateSpline

        # Extract data from file.
        subT,subV = read_an_port_file(self.fin,self.partsIx)
        return subV,subT
    
    def interpolate(self):
        """
        Interpolate velocity trajectory on record and replace self.t and self.v with linearly spaced
        interpolation.
        """
        from utils import MultiUnivariateSpline

        splineV = MultiUnivariateSpline(self.t,self.v)
        t = np.arange(self.t[0],self.t[-1],1/60)
        self.v = splineV(t)
        self.t = t
        # sync datetimes with linearly spaced seconds.
        self.tdate = np.array([self.tdate[0]+timedelta(0,i/60) for i in xrange(len(t))])
# end ANBroadcast
