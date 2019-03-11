# ================================================================================== #
# Module for reading and processing data from Axis Neuron broadcast port.
# Author: Eddie Lee edl56@cornell.edu
# ================================================================================== #

from .utils import *
from datetime import datetime,timedelta
import time
import socket
import shutil
from .axis_neuron import calc_file_headers
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
        print("Waiting for %s..."%start_file)
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

def _test_forward_timing(nIters=1000):
    """
    Test for the delay of forwarding a data point to a UDP port.
    """
    dt = np.zeros(nIters)
    stopEvent = threading.Event()

    try: 
        # Create and start a server UDP port on thread.
        servSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        def send_time():
            while not stopEvent.is_set():
                servSock.sendto(datetime.now().isoformat()+',',('127.0.0.1',7001))
                time.sleep(.01)
        servThread = threading.Thread(target=send_time)
        servThread.start()
        
        # Listen to server and record time.
        listenSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        listenSock.bind(('127.0.0.1',7001))
        for i in range(nIters):
            whenSent = listenSock.recv(54)
            whenRec = datetime.now()

            whenSent = whenSent.split(',')
            ix = np.where(np.array([len(s) for s in whenSent])==26)[0][0]
            if not ix is None:
                whenSent = whenSent[ix]
                whenSent = datetime.strptime(whenSent, '%Y-%m-%dT%H:%M:%S.%f')
                
                dt[i] = (whenRec-whenSent).total_seconds()
        stopEvent.set()
        servThread.join() 

    finally:
        servSock.close()
        listenSock.close()
    
    print("For %d samples, the delay is")
    print("Mean: %1.5f"%dt.mean())
    print("Min,max: %1.5f,%1.5f"%(dt.min(),dt.max()))
    print("Std: %1.5f"%dt.std())
    return dt

def record_AN_port(fname,port,
                   savedr=os.path.expanduser('~')+'/Dropbox/Sync_trials/Data',
                   host=HOST,
                   buffer_size=8192,
                   start_file='start',
                   stop_file='end'):
    """
    Start recording data from Axis Neuron port when presence of start is detected
    and stop when end_port_read is detected in DATADR.

    Parameters
    ----------
    fname : str
    port : int
    savedr : str,'~/Dropbox/Sync_trials/Data/'
    host : str,HOST
    start_file : str,'start'
    stop_file : str,'stop'
    """
    f = open('%s/%s'%(savedr,fname),'w')

    # Write header line.
    headers = calc_file_headers()[:-1]
    f.write('Timestamp, '+', '.join(headers)+'\n')

    # Check that recording has started as given by presence of lock file.
    while not os.path.isfile('%s/%s'%(savedr,start_file)):
        print("Waiting for start...")
        time.sleep(1)

    reader = ANReader(2,list(range(946)),port=port,host=host,port_buffer_size=buffer_size)
    try:
        reader.setup_port()
        while not os.path.isfile('%s/%s'%(savedr,stop_file)):
            v,t = reader.read_velocity()
            t = t.isoformat()

            f.write('%s, %s\n'%(t,str(v)[1:-1]))
    finally:
        reader.sock.close()
        f.close()

def load_AN_port(fname,dr='',time_as_dt=True,n_avatars=1,read_csv_kwargs={}):
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
    df.iloc[:,0] = df.iloc[:,0].apply(read_in_date)

    if time_as_dt:
        # Convert time stamp into time differences in seconds since first measurement.
        dt = np.cumsum( np.concatenate(([0],np.diff(df.iloc[:,0]).astype(int)/1e9)) )
        df['Timestamp'] = df['Timestamp'].apply(pd.to_numeric,errors='coerce')
        df['Timestamp'] = dt
    return df

def read_in_date(t_as_str):
    """
    t_as_str : str
    """
    try:
        return datetime.strptime(t_as_str, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(t_as_str, '%Y-%m-%dT%H:%M:%S')



# ======= #
# Classes #
# ======= #
class ANReader(object):
    def __init__(self,duration,parts_ix,
                 host=HOST,port=PORT,
                 port_buffer_size=8192,
                 max_buffer_size=1000,
                 recent_buffer_size=180,
                 always_empty_buffer=True,
                 verbose=False):
        """
        Class for reading and keeping track of data from Axis Neuron UDP port. UDP port reading is just
        saved to an array in memory which is fast, but the writing to disk is limited by disk access
        rates (much slower and unreliable).

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
            Dimension (n_time,3).
            Keeps record of self.duration seconds into the past. Only measures elapsed time excluding pauses.
        t : ndarray
            Keeps record of self.duration seconds into the past. Only measures elapsed time excluding pauses.
        vHistory : ndarray
            All history since last refresh corresponding to tHistory.
        tHistory : ndarray
            Elapsed time in seconds.
        vAsDateHistory : ndarray
            All history since last refresh corresponding to tAsDateHistory.
        tAsDateHistory : ndarray
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
        self.alwaysEmptyBuffer = always_empty_buffer
        self.verbose = verbose
        self.vAsDateHistory = []
        self.tAsDateHistory = []
        self.v = []
        self.tAsDate = []
    
    def __enter__(self):
        if self.verbose:
            print("Setting port up.")
        self.setup_port()

        # Start listening to port.
        if self.verbose:
            print("Listening")
        self.readThread = threading.Thread(target=self.listen_port)
        self.readThread.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.verbose:
            print("Initiating cleanup...")
        self.stopEvent.set()
        if self.verbose:
            print("Waiting for thread to join...")
        self.readThread.join()
        if self.verbose:
            print("Closing socket.")
        self.sock.close()

    def empty_buffer(self):
        """Empty buffer."""
        self.sock.setblocking(0)
        try:
            while True:
                self.sock.recv(self.portBufferSize)
        except:
            # When nothing is read from the buffer
            pass
        self.sock.setblocking(1)

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
        v = self.v[:]
        t = self.tAsDate[:]

        # Must be careful accessing these arrays because they're being asynchronously updated.
        # Fetch only the oldest overlapping data points.
        if len(v)==0 or len(t)==0:
            return np.array([]),np.array([]),np.array([])
        if len(v)<len(t):
            t = t[:len(v)]
        elif len(v)>len(t):
            v = v[:len(t)]
        assert len(v)==len(t)

        tAsDate = t
        t = np.cumsum([0.] + [dt.total_seconds() for dt in np.diff(t)])
        if interpolate and len(v)>10:
            t,v,tAsDate = self._interpolate(t,v,tAsDate)
        else:
            return np.array([]),np.array([]),np.array([])
        
        assert len(t)==len(v), "%d,%d"%(len(t),len(v))
        return np.array(v),np.array(t),np.array(tAsDate)

    def copy_history(self,interpolate=True):
        v = self.vAsDateHistory[:]
        t = self.tAsDateHistory[:]
        
        # Must be careful accessing these arrays because they're being asynchronously updated.
        # Fetch only the oldest overlapping data points.
        if len(v)==0 or len(t)==0:
            return np.array([]),np.array([]),np.array([]),np.array([])
        if len(v)<len(t):
            t = t[:len(v)]
        elif len(v)>len(t):
            v = v[:len(t)]
        assert len(t)==len(v) 

        tAsDate = t
        t = np.cumsum([0.] + [dt.total_seconds() for dt in np.diff(t)])
        if interpolate:
            if len(v)>10:
                t,v,tAsDate = self._interpolate(t,v,tAsDate)
            else:
                raise Exception("Can't interpolate")
        
        return np.array(v),np.array(t),np.array(tAsDate)
    
    # =============== # 
    # Data recording. #
    # =============== # 
    def setup_port(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("Trying to bind to host (%s,%d)"%(self.host,self.port))
        self.sock.bind((self.host,self.port))
        print("Bound.")
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
        
        if self.alwaysEmptyBuffer:
            self.empty_buffer()
        while len(rawData)<17200:
            rawData += self.sock.recv(self.portBufferSize)
            readTimes.append(datetime.now())
        rawData = rawData.split('\n')
        nBytes = [len(i) for i in rawData]
        
        # Take the longest read and split it by space delimiter.
        rawData = rawData[nBytes.index(max(nBytes))].split()
        if len(rawData)!=946:  # number of cols in calc file
            print("%d cols in calc output"%len(rawData))
            return []
        
        # Take time to the mean of the first and last read times. Read times seem to typically fall
        # between 10-30 ms.
        return (rawData,
                readTimes[0] + timedelta(seconds=(readTimes[-1]-readTimes[0]).total_seconds()/2.))

    def read_velocity(self):
        """
        Guarantee a data point from the port.
        
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
                    print("%s. Invalid float. Reading port again."%data[1].isoformat())
        return v,data[1]

    def listen_port(self):
        """Loop listening to the port."""
        # .pop and .append are atomic operations so they're thread safe.
        # http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
        # http://effbot.org/zone/thread-synchronization.htm#problems-with-simple-locking
        # https://stackoverflow.com/questions/6319207/are-lists-thread-safe
        while not self.stopEvent.is_set():
            v,t = self.read_velocity()

            if len(self.vAsDateHistory)==self.maxBufferSize:
                self.vAsDateHistory.pop(0)
                self.tAsDateHistory.pop(0)
            self.vAsDateHistory.append(v)
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
            print("Waiting for thread to join...")
        self.readThread.join()

        self.stopEvent.clear()
        
        self.lock.acquire()
        #while len(self.v)>0:
        #    self.v.pop(0)
        #    self.tAsDate.pop(0)
        #    self.vAsDateHistory.pop(0)
        #    self.tAsDateHistory.pop(0)
        self.v = []
        self.tAsDate = []
        self.vAsDateHistory = []
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
        from .utils import MultiUnivariateSpline
        from scipy.interpolate import interp1d
        assert len(t)==len(v)

        # Must be careful with knot spacing because the data frequency is highly variable.
        #splineV = MultiUnivariateSpline(t,v,fit_type='Uni')
        splineV = interp1d(t,v,axis=0,assume_sorted=True,copy=False)
        t = np.arange(t[0],t[-1],dt)
        v = splineV(t)
        t = t

        # sync datetimes with linearly spaced seconds.
        if not tdate is None:
            tdate = np.array([tdate[0]+timedelta(0,i*dt) for i in range(len(t))])
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
            print("Connection to %s:%d could not be established."%(self.host,self.port))
            return 
        
        # try to send data
        try:
            while not self.stopEvent.is_set():
                nBytesSent = sock.send(self._payload)
                    
                if verbose: print('%d bytes sent, %s'%(nBytesSent,self._payload))
                time.sleep(pause)  # if this pause goes immediately after connect, data transmission
                                   # is interrupted
        except:
            print("Connection closed unexpectedly.")
        finally:
            sock.close()
        
        print("DataBroadcaster thread stopped")
#end DataBroadcaster
