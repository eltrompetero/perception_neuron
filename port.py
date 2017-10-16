# ================================================================================== #
# Module for reading data from Axis Neuron broadcast port.
# Author: Eddie Lee edl56@cornell.edu
# ================================================================================== #
from __future__ import division
import numpy as np
from datetime import datetime,timedelta
import os
import time
import socket
import shutil
import pandas as pd
from load import calc_file_headers
import threading

HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = 7006  # Calculation data.
DATADR = os.path.expanduser('~')+'/Dropbox/Sync_trials/Data'



def forward_AN_port(ports,
                    host=HOST,
                    buffer_size=1024,
                    start_file='start_forwarding',
                    stop_file='stop_forwarding'):
    """
    Rebroadcast AN UDP broadcast from port 7006 to as many ports as given.
    
    Parameters
    ----------
    ports : list
    host : str,HOST
    buffer_size : int,1024
    start_file : str,'start_forwarding'
    stop_file : str,'stop_forwarding'
    """
    # Check that recording has started as given by presence of lock file.
    while not os.path.isfile('%s/%s'%(DATADR,start_file)):
        print "Waiting for %s..."%start_file
        time.sleep(1)
    
    try:
        listenSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        listenSock.bind((host,PORT))
        
        servSocks = [socket.socket(socket.AF_INET,socket.SOCK_DGRAM) for p in ports]

        while not os.path.isfile('%s/%s'%(DATADR,stop_file)):
            data = listenSock.recv(buffer_size)
            for p,sock in zip(ports,servSocks):
                sock.sendto(data,(host,p))
    finally:
        listenSock.close()
        [sock.close() for sock in servSocks]

def record_AN_port(fname,port,
                   savedr=os.path.expanduser('~')+'/Dropbox/Sync_trials/Data',
                   host=HOST,
                   start_file='start.txt',
                   stop_file='end_port_read.txt'):
    """
    Start recording data from Axis Neuron port when presence of start.txt is detected
    and stop when end_port_read.txt is detected in DATADR.

    Parameters
    ----------
    fname : str
    port : int
    savedr : str,'~/Dropbox/Sync_trials/Data/'
    host : str,HOST
    start_file : str,'start.txt'
    stop_file : str,'end_port_read.txt'
    """
    import platform

    f = open('%s/%s'%(savedr,fname),'w')

    # Write header line.
    headers = calc_file_headers()[:-1]
    f.write('Timestamp, '+', '.join(headers)+'\n')

    # Check that recording has started as given by presence of lock file.
    while not os.path.isfile('%s/%s'%(savedr,start_file)):
        print "Waiting for start.txt..."
        time.sleep(1)

    reader = ANReader(2,range(946),port=port,host=host,port_buffer_size=8000)
    try:
        reader.setup_port()
        while not os.path.isfile('%s/%s'%(savedr,stop_file)):
            v,t = reader.read_velocity()
            t = t.isoformat()

            f.write('%s, %s\n'%(t,str(v)[1:-1]))
    finally:
        reader.sock.close()
        f.close()

def load_AN_port(fname,dr='',time_as_dt=True,n_avatars=1,fix_file=True,read_csv_kwargs={}):
    """
    With data from a single individual at this moment.
    
    Parameters
    ----------
    fname : str
    dr : str,''
    time_as_dt : bool,True
    
    Returns
    -------
    df : pandas.DataFrame
    """
    if len(dr)>0:
        fname = '%s/%s'%(dr,fname)
    
    df = pd.read_csv(fname,**read_csv_kwargs)
    df.iloc[:,0] = df.iloc[:,0].apply(lambda t: datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f'))

    if time_as_dt:
        # Convert time stamp into time differences in seconds since first measurement.
        dt = np.cumsum( np.concatenate(([0],np.diff(df.iloc[:,0]).astype(int)/1e9)) )
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
    Get the stretch of avatar velocities that aligns with the velocity data of the subject. 

    Parameters
    ----------
    avatar : dict
        This would be the templateTrial loaded in VRTrial.
    part : str
        Choose from 'avatar','avatar0','hand','hand0'.
    t : array of datetime objects
        Stretch of time to return data from.
    t0 : datetime
    verbose : bool,False

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
    def __init__(self,duration,trial_type,parts_ix=None,broadcast_port=5001):
        """
        Parameters
        ----------
        duration : float
            Seconds into past to analyze for real time coherence measure.
        trial_type : str
            'avatar','avatar0','hand','hand0'
        parts_ix : list,None
            Indices of the columns in an_port file to extract and analyze. If None, this will be
            replaced automatically in start by the relevant hand.
        broadcast_port : int,5001
        """
        self.duration = duration
        self.trialType = trial_type
        self.partsIx = parts_ix
        self.broadcastPort = broadcast_port

    def load_avatar(self,return_subject=False):
        """
        Parameters
        ----------
        return_subject : bool,False

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
        
        if return_subject:
            subject = trial.subjectTrial
            return avatar,subject
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
        
        Code waits til when start_time.txt becomes available to read in avatar start time.
        
        The threads that are running:
        0. reader thread to read velocities from Axis Neuron UDP port.
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
        initial_vis_fraction : float,0.5
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
        coheval = CoherenceEvaluator(5,sample_freq=30,window_length=30)  # arg is max freq to consider
        gprmodel = GPR(tmin=min_window_duration,tmax=max_window_duration,
                       fmin=min_vis_fraction,fmax=max_vis_fraction)
        nextDuration = np.around(initial_window_duration,1)
        nextFraction = np.around(initial_vis_fraction,1)
        assert min_window_duration<=nextDuration<=max_window_duration
        assert min_vis_fraction<=nextFraction<=max_vis_fraction
        with open('%s/next_setting.txt'%DATADR,'w') as f:
            f.write('%1.1f,%1.1f'%(nextDuration,nextFraction))
        with open('%s/left_or_right.txt'%DATADR,'r') as f:
            handedness = f.read().rstrip()
        if handedness=='left':
            self.partsIx = left_hand_col_indices()
        else:
            self.partsIx = right_hand_col_indices()

        # Open port for communication with UE4 engine. This will send the current coherence value to
        # UE4.
        self.broadcast = DataBroadcaster(self.broadcastPort)
        self.broadcast.update_payload('0.00')
        broadcastThread = threading.Thread(target=self.broadcast.broadcast,
                                           kwargs={'pause':.5,'verbose':verbose})
        broadcastThread.start()

        # Set up thread for updating value of streaming broadcast of coherence.
        # This relies on reader.
        def update_broadcaster(reader,stopEvent):
            try:
                while not stopEvent.is_set():
                    v,t,tAsDate = reader.copy_recent()
                    if len(v)>=(self.duration*30*.95):
                        avv = fetch_matching_avatar_vel(avatar,self.trialType,tAsDate,t0)
                        v = fetch_matching_avatar_vel(subject,self.trialType,tAsDate,t0)
                        avgcoh = coheval.evaluateCoherence( avv[:,2],v[:,2] )
                        #v = np.random.random(size=avv.shape)
                        #avgcoh = np.abs( (avv*v).sum(1) / 
                        #                 (np.linalg.norm(avv,axis=1)*np.linalg.norm(v,axis=1))
                        #               ).mean()
                        print "new coherence is %1.2f"%avgcoh
                        self.broadcast.update_payload('%1.2f'%avgcoh)
                    time.sleep(0.1)
            finally:
                print "updateBroadcastThread stopped"
        self.updateBroadcastEvent = threading.Event()

        # Wait til start_time.txt has been written to start experiment..
        t0 = self.wait_for_start_time()
        avatar,subject = self.load_avatar(return_subject=True)

        # Wait til fully visible trial has finished and read data while waiting so that we can erase
        # it before starting the next trial.
        time.sleep(self.duration+1)
        self.wait_for_start_gpr()
        
        if verbose:
            print "Starting threads."
        with ANReader(self.duration,self.partsIx,
                      port=7011,
                      verbose=True,
                      port_buffer_size=8000,
                      recent_buffer_size=self.duration*60) as reader:
            
            updateBroadcastThread = threading.Thread(target=update_broadcaster,
                                                     args=(reader,self.updateBroadcastEvent))

            while reader.len_history()<(self.duration*60):
                if verbose:
                    print "Waiting to collect more data...(%d)"%reader.len_history()
                self.broadcast.update_payload('0.00')
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
                        self.broadcast.update_payload('0.00')
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
                 host=HOST,port=PORT,
                 port_buffer_size=1024,
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
        parts_ix : list of ints
        host : str,HOST
        port : int,PORT
        broadcast_file : str
            Where data from AN port is broadcast.
        parts_ix : list of ints
            Indices of corresponding column headers in broadcast file.
        port_buffer_size : int,1024
            On Mac OS, this is the default buffer size for AN as of v3.8.42.6503. For Windows, you
            must adjust this to be larger than 6000 (at least) because the buffer size is bigger.
            Otherwise messages will be dropped.
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
        self.host = host
        self.port = port
        self.portBufferSize = port_buffer_size
        self.maxBufferSize = max_buffer_size
        self.recentBufferSize = recent_buffer_size
        self.verbose = verbose
        self.vHistory = []
        self.tAsDateHistory = []
        self.v = []
        self.tAsDate = []
    
    def __enter__(self):
        if self.verbose:
            print "Setting port up."
        self.setup_port()

        # Start listening to port.
        if self.verbose:
            print "Listening"
        self.readThread = threading.Thread(target=self.listen_port)
        self.readThread.start()

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

        tAsDate = t
        t = np.cumsum([0.] + [dt.total_seconds() for dt in np.diff(t)])
        if interpolate and len(v)>10:
            t,v,tAsDate = self._interpolate(t,v,tAsDate)
        else:
            self.lock.release()
            return np.array([]),np.array([]),np.array([])
        
        self.lock.release()
        assert len(t)==len(v), "%d,%d"%(len(t),len(v))
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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print "trying to bind to host %s"%self.host
        self.sock.bind((self.host,self.port))
        self.rawData = []

    def read_port(self):
        """
        Read port and return something if a data point was returned.

        Returns
        -------
        Empty list if no data, otherwise tuple of data and read times.
        """
        from datetime import timedelta
        
        rawData = ''
        readTimes = []
        while len(rawData)<18000:
            rawData += self.sock.recv(self.portBufferSize)
            readTimes.append(datetime.now())
        rawData = rawData.split('\n')
        nBytes = [len(i) for i in rawData]
        
        rawData = rawData[nBytes.index(max(nBytes))].split()
        if len(rawData)!=946:  # number of cols in calc file
            print "%d cols in calc output"%len(rawData)
            return []
        
        # Take time to the mean of the first and last read times. Read times seem to
        # typically fall between 10-30 ms.
        return rawData, readTimes[0] + timedelta( seconds=(readTimes[-1]-readTimes[0]).total_seconds()/2. )

    def read_velocity(self):
        """
        Get a data point from the port.
        
        Returns
        -------
        v : list
        timestamp : datetime.datetime
        """
        v = []
        while len(v)==0:
            data = self.read_port()
            try:
                if len(data)>0:
                    v = [float(data[0][ix]) for ix in self.partsIx]
            except ValueError:
                    print "%s. Invalid float. Reading port again."%data[1].isoformat()
            # This sleep time works well on the Mac. Almost always returns a buffer of
            # 946 data points.
            #time.sleep(.01)
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
    def _interpolate(self,t,v,tdate=None,dt=1/30):
        """
        Interpolate velocity trajectory on record and replace self.t and self.v with linearly spaced
        interpolation.

        Parameters
        ----------
        t : ndarray
        v : ndarray
        tdate : ndarray,None
        dt : float,1/30
            Time spacing for interpolation.
        """
        from utils import MultiUnivariateSpline
        assert len(t)==len(v)
        
        # Must be careful with knot spacing because the data frequency is highly variable.
        splineV = MultiUnivariateSpline(t,v,fit_type='Uni')
        t = np.arange(t[0],t[-1],dt)
        v = splineV(t)
        t = t

        # sync datetimes with linearly spaced seconds.
        if not tdate is None:
            tdate = np.array([tdate[0]+timedelta(0,i*dt) for i in xrange(len(t))])
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
        self._payload = snew

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
        # try to connect to port
        try:
            sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            sock.connect((self.host,self.port))
        except:
            print "Connection to %s:%d could not be established."%(self.host,self.port)
            return 
        
        # try to send data
        try:
            while not self.stopEvent.is_set():
                nBytesSent = sock.send(self._payload)
                    
                if verbose: print '%d bytes sent, %s'%(nBytesSent,self._payload)
                time.sleep(pause)  # if this pause goes immediately after connect, data transmission
                                   # is interrupted
        finally:
            sock.close()
        
        print "DataBroadcaster thread stopped"
#end DataBroadcaster
