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
PORT = 7006  # Calculation data.
DATADR = os.path.expanduser('~')+'/Dropbox/Sync_trials/Data'

# Functions for reading from broadcasting port.
def read_port():
    def incoming(host, port):
        """Open specified port and return file-like object"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        sock.connect((host,port))
        # In worst case scenario, all 947 fields will be 9 bytes with 1 space in between and a \n
        # character , but AN only returns 1024 bytes at a time.
        t,data = datetime.now(),sock.recv(1024)
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
    NOTE: Need two threads. One to read from port and other to save to file.
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
def read_an_port_file(fopen,partsIx,verbose=False):
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
    # The first time the file is opened, the read position is at the beginning of the file.
    # The ANReader instance should open the file and seek to the end before calling this function.
    fopen.seek(0,1)
    ln = fopen.read() 
    # Take last line. 
    ln = ln.split('\n')[:-1]
    if len(ln)<2:
        return subT,subV
    
    if verbose:
        print "Accessing file..."
    for el in ln:
        # Some values seem to be cutoff because microseconds is precisely 0.
        el = el.split()
        if el[0]!='Timestamp' and len(el)==947:
            if len(el[0])!=26 :
                el[0] += '.000000'
            
            # Read in data. Sometimes there seems to problems with the data that as been read in.
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

def fetch_matching_avatar_vel(avatar,part,t,t0,verbose=False):
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
    if verbose:
        print "Getting avatar times between %1.1fs and %1.1fs."%(t[0],t[-1])

    # Return part of avatar's trajectory that agrees with the stipulated time bounds.
    return avatar[part+'V'](t)



# ======= #
# Classes #
# ======= #
class HandSyncExperiment(object):
    def __init__(self,duration,trial_type,parts_ix,broadcast_port=5001):
        """
        Parameters
        ----------
        duration : float
            Seconds into past to analyze for real time coherence measure.
        trial_type : str
            'avatar','avatar0','hand','hand0'
        parts_ix : str
            Indices of the columns in an_port file to extract and analyze.
        broadcast_port : int,5001
        """
        self.duration = duration
        self.trialType = trial_type
        self.partsIx = parts_ix
        self.broadcastPort = broadcast_port

    def load_avatar(self):
        """
        Returns
        -------
        avatar : dict
            Dictionary of avatar interpolation splines.
        """
        from load import subject_settings_v3 as subject_settings
        from load import VRTrial
        handedness = open('%s/%s'%(DATADR,'left_or_right.txt')).readline().rstrip()

        if handedness=='left':
            person,modelhandedness,rotation,dr = subject_settings(0)
            self.partsIx = left_hand_col_indices()
        elif handedness=='right':
            person,modelhandedness,rotation,dr = subject_settings(2)
            self.partsIx = right_hand_col_indices()
        else:
            print handedness
            raise Exception

        trial = VRTrial(person,modelhandedness,rotation,dr)
        avatar = trial.templateTrial

        return avatar

    def wait_for_start_time(self):
        """
        Get the time at which the trial started.

        Returns
        -------
        t0 : datetime
            The time at which the trial was started.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start_time.txt')):
            time.sleep(.5)
        # Give some time for the initial time to be written by UE4.
        time.sleep(.5)
        with open('%s/%s'%(DATADR,'start_time.txt'),'r') as f:
            t0 = datetime.strptime(f.readline(),'%Y-%m-%dT%H:%M:%S.%f')
        return t0
   
    def wait_for_start_gpr(self):
        """
        Once start_gpr.txt has been written, erase self.subVBroadcast's memory of the history of
        velocity.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start_gpr.txt')):
            time.sleep(.5)

    def start(self,
              update_delay=.3,
              initial_window_duration=1.0,initial_vis_fraction=0.5,
              min_window_duration=.6,max_window_duration=2,
              min_vis_fraction=.1,max_vis_fraction=.9,
              verbose=False):
        """
        Run realtime analysis for experiment.
        
        Code waits til when start.txt becomes available to read in avatar start time.
        
        The threads that are running:
        0. readThread: read an_port.txt to get the latest velocity data
        1. updateBroadcastThread: assess subject's performance relative to the avatar and
            update coherence value
        2. broadcastThread: broadcast subject's performance to port 5001
        
        Thread communication happens through members that are updated using thread locks.
        
        In while loop, run GPR prediction step and write the next window duration and visible
        fraction to file. Waiting for run_gpr.txt and writing to next_setting.txt.

        When end.txt is written, experiment ends.

        NOTE:
        - only calculate coherence along z-axis
        
        Parameters
        ----------
        update_delay : float,.3
            Number of seconds to wait between updating arrays when calculating realtime coherence.
        initial_window_duration : float,1.0
        min_window_duration : float,.5
        max_window_duration : float,2
        min_vis_fraction : float,.1
        max_vis_fraction : float,.9
        verbose : bool,False
        """
        from load import subject_settings_v3,VRTrial
        from gpr import CoherenceEvaluator,GPR
        import cPickle as pickle
        
        # Setup routines for calculating coherence.
        coheval = CoherenceEvaluator(10)  # arg is max freq to consider
        gprmodel = GPR(tmin=min_window_duration,tmax=max_window_duration,
                       fmin=min_vis_fraction,fmax=max_vis_fraction)
        nextDuration = np.around(initial_window_duration,1)
        nextFraction = np.around(initial_vis_fraction,1)
        assert min_window_duration<=nextDuration<=max_window_duration
        assert min_vis_fraction<=nextFraction<=max_vis_fraction
        with open('%s/next_setting.txt'%DATADR,'w') as f:
            f.write('%1.1f,%1.1f'%(nextDuration,nextFraction))

        # Open port for communication with UE4 engine. This will send the current coherence value to
        # UE4.
        self.broadcast = DataBroadcaster(self.broadcastPort)
        self.broadcast.update_payload('0.0')
        broadcastThread = threading.Thread(target=self.broadcast.broadcast,kwargs={'verbose':False})
        broadcastThread.start()

        # Wait til start_time.txt has been written to start experiment..
        t0 = self.wait_for_start_time()
        avatar = self.load_avatar()

        # Wait til fully visible trial has finished and read data while waiting so that we can erase
        # it before starting the next trial.
        time.sleep(self.duration+1)
        self.wait_for_start_gpr()
        
        if verbose:
            print "Starting threads."
        with ANReader(self.duration,self.partsIx,verbose=True) as reader:
            # Set up thread for updating value of streaming braodcast of coherence.
            def update_broadcaster(stopEvent):
                try:
                    while not stopEvent.is_set():
                        v,t,tAsDate = reader.copy_recent()
                        if len(v)>=160:
                            avv = fetch_matching_avatar_vel(avatar,self.trialType,tAsDate,t0)
                            avgcoh = coheval.evaluateCoherence( avv[:,2],v[:,2] )
                            self.broadcast.update_payload('%1.1f'%avgcoh)
                        time.sleep(0.5)
                finally:
                    print "updateBroadcastThread stopped"
            self.updateBroadcastEvent = threading.Event()
            updateBroadcastThread = threading.Thread(target=update_broadcaster,args=(self.updateBroadcastEvent,))

            while reader.len_history()<(self.duration*60):
                if verbose:
                    print "Waiting to collect more data...(%d)"%reader.len_history()
                time.sleep(1.5)
            if self.broadcast.connectionInterrupted:
                raise Exception
            updateBroadcastThread.start()
            
            # Run GPR for the next windows setting.
            while not os.path.isfile('%s/%s'%(DATADR,'end.txt')):
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
                    v,t,tdateHistory = reader.copy_history()
                    avv = fetch_matching_avatar_vel(avatar,self.trialType,
                                                    tdateHistory,t0)
                    avgcoh = coheval.evaluateCoherence( avv[:,2],v[:,2] )
                    nextDuration,nextFraction = gprmodel.update( avgcoh,nextDuration,nextFraction )
                    open('%s/next_setting.txt'%DATADR,'w').write('%1.1f,%1.1f'%(nextDuration,
                                                                                nextFraction))

                    # Stop thread reading from port and refresh history.
                    reader.refresh()
                    while reader.len_history()<(self.duration*60):
                        if verbose:
                            print "Waiting to collect more data...(%d)"%reader.len_history()
                        time.sleep(1.5)
                    
                time.sleep(update_delay) 
            
        # Always end thread.
        print "Ending threads..."
        self.stop()
        updateBroadcastThread.join()
        broadcastThread.join()

        with open('%s/%s'%(DATADR,'end_port_read.txt'),'w') as f:
            f.write('')

        print "Saving GPR."
        pickle.dump({'gprmodel':gprmodel},open('%s/%s'%(DATADR,'temp.p'),'wb'),-1)

    def stop(self):
        """Stop all thread that could be running. This does not wait for threads to stop."""
        self.updateBroadcastEvent.set()
        self.broadcast.stopEvent.set()
# end HandSyncExperiment


class ANReader(object):
    def __init__(self,duration,parts_ix,
                 port_buffer_size=9460,
                 max_buffer_size=1000,
                 recent_buffer_size=180,
                 verbose=False):
        """
        Class for reading from Axis Neuron UDP port and writing to file. UDP port reading is just
        saved to an array in memory which is fast, but the writing to disk is limited by disk access
        rates (much slower and unreliable).
        This will also keeping track of history of velocity.
        
        Parameters
        ----------
        duration : float
            Number of seconds to keep track in the past.
        broadcast_file : str
            Where data from AN port is broadcast.
        parts_ix : list of ints
            Indices of corresponding column headers in broadcast file.
        port_buffer_size : int,9460
        max_buffer_size : int,1000
            Number of port calls to keep in memory at any given time.
        recent_buffer_size : int,180
        verbose : bool,False

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
        self.stopEvent = threading.Event()
        self.readCondition = threading.Condition()
        
        if type(parts_ix) is int:
            parts_ix = [parts_ix]
        self.partsIx = parts_ix
        self.portBufferSize = port_buffer_size
        self.maxBufferSize = max_buffer_size
        self.recentBufferSize = recent_buffer_size
        self.verbose = verbose
        self.vHistory = []
        self.tAsDateHistory = []
        self.v = []
        self.tAsDate = []

        print "done setting up"
    
    def __enter__(self):
        if self.verbose:
            print "Setting port up."
        self.setup_port()

        # Start listening to port.
        if self.verbose:
            print "Listening"
        self.readThread = threading.Thread(target=self.listen_port)
        self.readThread.start()

        #if write:
        #    self.write_to_file()
        #
        #self.partsIx = parts_ix
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.verbose:
            print "Initiating cleanup..."
        self.stopEvent.set()
        if self.verbose:
            print "Waiting for thread to join..."
        self.readThread.join()
        if self.verbose:
            print "Closing socket."
        self.sock.close()

    # ========================= # 
    # Safe data access methods. #
    # ========================= # 
    def len_history(self):
        return len(self.tAsDateHistory)

    def copy_recent(self,interpolate=True):
        """
        Parameters
        ----------
        interpolate : bool,True

        Returns
        -------
        v : ndarray
        t : ndarray
        tAsDate : ndarray
        """
        self.lock.acquire()

        # Must be careful accessing these arrays because they're being asynchronously updated.
        v = self.v[:]
        t = self.tAsDate[:]
        if len(v)==0 or len(t)==0:
            self.lock.release()
            return np.array([]),np.array([]),np.array([])
        if len(v)<len(t):
            t = t[:len(t)-len(v)]
        elif len(v)>len(t):
            v = v[:len(v)-len(t)]
        assert len(t)==len(v), "%d,%d"%(len(t),len(v))

        tAsDate = t
        t = np.cumsum([0.] + [dt.total_seconds() for dt in np.diff(t)])
        if interpolate and len(v)>10:
            t,v,tAsDate = self._interpolate(t,v,tAsDate)
        else:
            self.lock.release()
            return np.array([]),np.array([]),np.array([])
        
        self.lock.release()
        return np.array(v),np.array(t),np.array(tAsDate)

    def copy_history(self,interpolate=True):
        self.lock.acquire()
        v = self.vHistory[:]
        t = self.tAsDateHistory[:]
        
        # Must be careful accessing these arrays because they're being asynchronously updated.
        v = self.v[:]
        t = self.tAsDate[:]
        if len(v)==0 or len(t)==0:
            self.lock.release()
            return np.array([]),np.array([]),np.array([]),np.array([])
        if len(v)<len(t):
            t = t[:len(t)-len(v)]
        elif len(v)>len(t):
            v = v[:len(v)-len(t)]
        assert len(t)==len(v) 

        tAsDate = t
        t = np.cumsum([0.] + [dt.total_seconds() for dt in np.diff(t)])
        if interpolate and len(v)>10:
            t,v,tAsDate = self._interpolate(t,v,tAsDate)
        else:
            raise Exception("Can't interpolate")
        
        self.lock.release()
        return np.array(v),np.array(t),np.array(tAsDate)
    
    # =============== # 
    # Data recording. #
    # =============== # 
    def setup_port(self):
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.bind((HOST,PORT))
        self.rawData = []

    def read_port(self):
        """
        Read port and return something if a data point was returned.

        Returns
        -------
        Empty list if no data, otherwise tuple of data and read times.
        """
        rawData = ''
        readTimes = []
        while len(rawData)<18000:
            rawData += self.sock.recv(self.portBufferSize)
            readTimes.append(datetime.now())
        rawData = rawData.split('\n')
        nBytes = [len(i) for i in rawData]

        rawData = rawData[nBytes.index(max(nBytes))].split()
        if len(rawData)!=946:  # number of cols in calc file
            return []
        return rawData,readTimes[nBytes.index(max(nBytes))]

    def read_velocity(self):
        """
        Get a data point from the port.
        """
        v = []
        while len(v)==0:
            data = self.read_port()
            if len(data)>0:
                v = [float(data[0][ix]) for ix in self.partsIx]
        return v,data[1]

    def listen_port(self):
        """Loop listening to the port."""
        # .pop and .append are atomic operations so they're thread safe.
        # http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
        # http://effbot.org/zone/thread-synchronization.htm#problems-with-simple-locking
        # https://stackoverflow.com/questions/6319207/are-lists-thread-safe
        while not self.stopEvent.is_set():
            v,t = self.read_velocity()

            if len(self.vHistory)==self.maxBufferSize:
                self.vHistory.pop(0)
                self.tAsDateHistory.pop(0)
            self.vHistory.append(v)
            self.tAsDateHistory.append(t)

            if len(self.v)==self.recentBufferSize:
                self.v.pop(0)
                self.tAsDate.pop(0)
            self.v.append(v)
            self.tAsDate.append(t)

    # ================== #
    # Interface methods. #
    # ================== #
    def refresh(self):
        """Clear stored arrays including history."""
        self.stopEvent.set()
        if self.verbose:
            print "Waiting for thread to join..."
        self.readThread.join()

        self.stopEvent.clear()
        
        self.lock.acquire()
        #while len(self.v)>0:
        #    self.v.pop(0)
        #    self.tAsDate.pop(0)
        #    self.vHistory.pop(0)
        #    self.tAsDateHistory.pop(0)
        self.v = []
        self.tAsDate = []
        self.vHistory = []
        self.tAsDateHistory = []
        self.lock.release()

        # Start listening to port.
        self.readThread = threading.Thread(target=self.listen_port)
        self.readThread.start()
    
    # ======== #
    # Private. #
    # ======== #
    def _interpolate(self,t,v,tdate=None):
        """
        Interpolate velocity trajectory on record and replace self.t and self.v with linearly spaced
        interpolation.

        Parameters
        ----------
        t : ndarray
        v : ndarray
        tdate : ndarray,None
        """
        from utils import MultiUnivariateSpline
        assert len(t)==len(v)
        
        # Must be careful with knot spacing because the data frequency is highly variable.
        splineV = MultiUnivariateSpline(t,v,fit_type='Uni')
        t = np.arange(t[0],t[-1],1/60)
        v = splineV(t)
        t = t

        # sync datetimes with linearly spaced seconds.
        if not tdate is None:
            tdate = np.array([tdate[0]+timedelta(0,i/60) for i in xrange(len(t))])
            return t,v,tdate
        return t,v
# end ANReader



class DataBroadcaster(object):
    def __init__(self,port,host='127.0.0.1'):
        """
        Class for safely broadcasting mutable data.

        Parameters
        ----------
        port : int
        host : str,'127.0.0.1'
        """
        self.port = port
        self.host = host
        self.lock = threading.Lock()
        self.stopEvent = threading.Event()
        self._payload = ''
        self.connectionInterrupted = False

    def update_payload(self,snew):
        """
        Update string of data that is being transmitted.
        
        Parameters
        ----------
        snew : str
        """
        self.lock.acquire()
        self._payload = snew
        self.lock.release()

    def broadcast(self,pause=1,verbose=False):
        """
        Loop broadcast payload to port with pause in between broadcasts. Payload is locked so
        that it can be updated safely from another function while this loop is running.

        self.connectionInterrupted keeps track of whether or not connection with server was closed
        by client or not.

        Parameters
        ----------
        pause : float,1
            Number of seconds to wait before looping.
        verbose : bool,False
        """
        while not self.stopEvent.is_set():
            try:
                sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                sock.connect((self.host,self.port))

                self.lock.acquire()
                nBytesSent = sock.send(self._payload)
                self.lock.release()
                
                if verbose: print '%d bytes sent'%nBytesSent

                sock.close()
                time.sleep(pause)  # if this pause goes immediately after connect, data transmission
                                   # is interrupted
            except:
                print "Error: Connection to server broken."
                self.connectionInterrupted = True
                break
        print "DataBroadcaster thread stopped"
        self.connectionInterrupted = False
#end DataBroadcaster
