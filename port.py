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
import threading

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

def record_AN_port(fname,savedr='C:/Users/Eddie/Dropbox/Sync_trials/Data'):
    """
    Start recording data from Axis Neuron port when presence of start.txt is detected and stop when
    end_port_read.txt is detected in DATADR.

    Must be run on Windows running the UE4 setup.

    Parameters
    ----------
    fname : str
    savedr : str,'C:/Users/Eddie/Dropbox/Sync_trials/Data/'
    """
    f = open(fname,'w')

    # Check that recording has started as given by presence of lock file.
    while not os.path.isfile('%s/%s'%(savedr,'start.txt')):
        time.sleep(1)
    
    # Write header line.
    headers = list(calc_file_headers())
    headers[-1] = ''.join(headers[-1].split())  # Remove space in last column header.
    f.write('Timestamp '+' '.join(headers)+'\n')
    
    # Capture port output.
    while not os.path.isfile('%s/%s'%(savedr,'end_port_read.txt')):
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
    if len(ln)<2:
        return subT,subV
    
    print "Accessing file..."
    for el in ln:
        # Some values seem to be cutoff because microseconds is precisely 0.
        el = el.split()
        if el[0]!='Timestamp':
            if len(el[0])!=26 :
                el[0] += '.000000'
            
            # Read in data.
            subT = np.append(subT,datetime.strptime(el[0], '%Y-%m-%dT%H:%M:%S.%f'))
            subV = np.append(subV,[[float(f) for f in [el[ix] for ix in partsIx]]],axis=0)
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

def fetch_matching_avatar_vel(avatar,part,t,t0,disp=False):
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
    t0 : datetime

    Returns
    -------
    v : ndarray
        Avatar's velocity that matches given time stamps.
    """
        
    # Transform dt to time in seconds.
    t = np.array([i.total_seconds() for i in t-t0])
    if disp:
        print "Getting avatar times between %1.1fs and %1.1fs."%(t[0],t[-1])

    # Return part of avatar's trajectory that agrees with the stipulated time bounds.
    return avatar[part+'V'](t)



# ======= #
# Classes #
# ======= #
class HandSyncExperiment(object):
    def __init__(self,outfile,duration,trial_type):
        """
        Parameters
        ----------
        outfile : str
            Output file where to write current coherence value. This will be placed in DATADR.
        duration : float
            Seconds into past to analyze for real time coherence measure.
        trial_type : str
            'avatar','avatar0','hand','hand0'
        parts_ix : str
            Indices of the columns in an_port file to extract and analyze.
        """
        self.outfile = outfile
        self.duration = duration
        self.trialType = trial_type
        self.partsIx = None

    def load_avatar(self):
        """
        Returns
        -------
        avatar : dict
            Dictionary of avatar interpolation splines.
        """
        from load import subject_settings_v3 as subject_settings
        from load import VRTrial
        handedness = open('%s/%s'%(DATADR,'left_or_right.txt')).readline()

        if handedness=='left':
            person,modelhandedness,rotation,dr = subject_settings(0)
            self.partsIx = left_hand_col_indices()
        elif handedness=='right':
            person,modelhandedness,rotation,dr = subject_settings(2)
            self.partsIx = right_hand_col_indices()
        else:
            raise Exception

        trial = VRTrial(person,modelhandedness,rotation,dr)
        avatar = trial.templateTrial

        return avatar

    def read_start_time(self):
        """
        Get the time at which the trial started.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start_time.txt')):
            time.sleep(.5)
        time.sleep(.5)
        with open('%s/%s'%(DATADR,'start_time.txt'),'r') as f:
            t0 = datetime.strptime(f.readline(),'%Y-%m-%dT%H:%M:%S.%f')
        return t0
   
    def wait_for_start_gpr(self):
        while not os.path.isfile('%s/%s'%(DATADR,'start_gpr.txt')):
            self.subVBroadcast.update()
            time.sleep(.5)
        self.subVBroadcast.refresh()

    def start(self,
              update_delay=.3,
              initial_window_duration=1.0,initial_vis_fraction=0.5,
              min_window_duration=.5,max_window_duration=2,
              min_vis_fraction=.1,max_vis_fraction=.9):
        """
        Run realtime analysis for experiment. Starts when start.txt becomes available to read in
        avatar start time. Reads an_port.txt to get the latest velocity data and keep track of
        subject's performance relative to the avatar. When run_gpr.txt is written, GPR prediction
        for the coherence as a function of window duration and visible fraction is performed and the
        next coordinate is written to next_setting.txt.

        Parameters
        ----------
        update_delay : float,.3
            Number of seconds to wait between updating arrays when calculating realtime coherence.
        initial_window_duration : float,1.0
        min_window_duration : float,.5
        max_window_duration : float,2
        min_vis_fraction : float,.1
        max_vis_fraction : float,.9
        """
        from load import subject_settings_v3,VRTrial
        from gpr import CoherenceEvaluator,GPR
        import cPickle as pickle
        
        # Setup routines for calculating coherence.
        ceval = CoherenceEvaluator(10)
        gprmodel = GPR(TMIN=min_window_duration,TMAX=max_window_duration,
                       FMIN=min_vis_fraction,FMAX=max_vis_fraction)
        nextDuration = np.around(initial_window_duration,1)
        nextFraction = np.around(initial_vis_fraction,1)
        assert min_window_duration<=nextDuration<=max_window_duration
        assert min_vis_fraction<=nextFraction<=max_vis_fraction
        open('%s/next_setting.txt'%DATADR,'w').write('%1.1f,%1.1f'%(nextDuration,nextFraction))
        
        t0 = self.read_start_time()
        avatar = self.load_avatar()

        # For retrieving the subject's velocities.
        self.subVBroadcast = ANBroadcast(self.duration,
                                         '%s/%s'%(DATADR,'an_port.txt'),
                                         self.partsIx)

        # Wait til fully visible trial has finished and read data while waiting so that we can erase
        # it before starting the next trial.
        time.sleep(self.duration+1)
        self.wait_for_start_gpr()
        
        event = threading.Event()

        # Run real time GPR analysis loop.
        with open('%s/%s'%(DATADR,self.outfile),'w') as fout:
            # Start thread for looping update subVBroadcast.
            t = threading.Thread(target=self.subVBroadcast.loop_update,args=(event,update_delay))
            t.setDaemon(True)
            t.start()
            
            while self.subVBroadcast.len_history()<(self.duration*60):
                print "Waiting to collect more data...(%d)"%self.subVBroadcast.len_history()
                time.sleep(1.5)
            
            while not os.path.isfile('%s/%s'%(DATADR,'end.txt')):
                v = self.subVBroadcast.copy_v()
                avv = fetch_matching_avatar_vel(avatar,self.trialType,self.subVBroadcast.copy_tdate(),
                                                t0,
                                                disp=True)
                if len(v)>90:
                    avgcoh = ceval.evaluateCoherence(avv[:,2],v[:,2])
                    fout.write('%f\n'%avgcoh)
                print "loop"     
                if os.path.isfile('%s/%s'%(DATADR,'run_gpr.txt')):
                    # Sometimes deletion conflicts with writing.
                    notDeleted = True
                    while notDeleted:
                        try:
                            os.remove('%s/%s'%(DATADR,'run_gpr.txt'))
                            notDeleted = False
                            print "run_gpr.txt successfully deleted."
                        except OSError:
                            print "run_gpr.txt unsuccessfully deleted."
                            time.sleep(.1)

                    # Run GPR.
                    print "Running GPR on this trial..."
                    avv = fetch_matching_avatar_vel(avatar,self.trialType,
                                                    self.subVBroadcast.copy_tdateHistory(),t0)
                    avgcoh = ceval.evaluateCoherence( avv[:,2],self.subVBroadcast.copy_vHistory()[:,2] )
                    nextDuration,nextFraction = gprmodel.update( avgcoh,nextDuration,nextFraction )
                    open('%s/next_setting.txt'%DATADR,'w').write('%1.1f,%1.1f'%(nextDuration,
                                                                                nextFraction))

                    # Refresh history.
                    event.set()
                    self.subVBroadcast.refresh()
                    event.clear()

                    # No output til more data has been collected.
                    print "Collecting data..."
                    time.sleep(self.duration+1)
                    
                    # Start loop update again.
                    t = threading.Thread(target=self.subVBroadcast.loop_update,args=(event,update_delay))
                    t.start()

                time.sleep(update_delay) 
            
            # Always end thread.
            print "ending thread"
            event.set()
            t.join()

            with open('%s/%s'%(DATADR,'end_port_read.txt'),'w') as f:
                f.write('')

        print "Saving GPR."
        pickle.dump({'gprmodel':gprmodel},open('%s/%s'%(DATADR,'temp.p'),'wb'),-1)
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

        Subfields
        ---------
        v : ndarray
            Of duration of self.duration. Of dimension (n_time,3).
        t : ndarray
            Of duration of self.duration.
        vHistory : ndarray
            All history since last refresh.
        tHistory : ndarray
            All history since last refresh.
        tdateHistory : ndarray
            All history since last refresh.
        lock : threading.Lock

        Methods
        -------
        refresh()
        update()
        fetch_new_vel()
        _interpolate()
        """
        self.duration = duration
        self.lock = threading.Lock()
        
        # Open file and go to the end.
        self.fin = open(broadcast_file)
        self.fin.seek(0,2)
        
        self.partsIx = parts_ix

        self.refresh()

    # ========================= # 
    # Safe data access methods. #
    # ========================= # 
    def len_history(self):
        self.lock.acquire()
        n = len(self.tdateHistory)
        self.lock.release()
        return n

    def copy_v(self):
        self.lock.acquire()
        v = self.v.copy()
        self.lock.release()
        return v

    def copy_tdate(self):
        self.lock.acquire()
        tdate = self.tdate.copy()
        self.lock.release()
        return tdate

    def copy_v(self):
        self.lock.acquire()
        v = self.v.copy()
        self.lock.release()
        return v

    def copy_vHistory(self):
        self.lock.acquire()
        vHistory = self.vHistory.copy()
        self.lock.release()
        return vHistory

    def copy_tdateHistory(self):
        self.lock.acquire()
        tdateHistory = self.tdateHistory.copy()
        self.lock.release()
        return tdateHistory

    # =============== #
    # Update methods. #
    # =============== #
    def refresh(self):
        """Clear stored arrays including history."""
        self.lock.acquire()

        self.v = np.zeros((0,3))
        self.t = np.array(())
        self.tdate = np.array(())
        self.vHistory = np.zeros((0,3))
        self.tHistory = np.array(())
        self.tdateHistory = np.array(())

        self.lock.release()

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
        
        self.lock.acquire()

        # Update arrays with new data. 
        self.vHistory = np.append(self.vHistory,vnew,axis=0)
        self.tdateHistory = np.append(self.tdateHistory,tdatenew)
        now = datetime.now()
        self.tHistory = np.array([(t-now).total_seconds() for t in self.tdateHistory])
        
        # Only keep data points that are within self.duration of now.
        tix = self.tHistory>-self.duration
        self.v = self.vHistory[tix]
        self.t = self.tHistory[tix]
        self.tdate = self.tdateHistory[tix]
        assert len(self.v)==len(self.t)==len(self.tdate) 
        
        # Must have enough points to interpolate.
        if len(self.v)>10:
            self._interpolate()

        self.lock.release()

    def loop_update(self,event,dt):
        """
        Loop update til event is set.

        Parameters
        ----------
        event : threading.Event
        dt : float
            Number of seconds to wait between updates. Must be greater than 0.3.
        """
        assert dt>=.3,"Update loop too fast for acquiring outside of function."
        while not event.is_set():
            self.update()
            time.sleep(dt)
    
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
    
    def _interpolate(self):
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
