# ==================================================================== #
# Class for running synchronization experiments.
# Author: Eddie Lee edl56@cornell.edu
# ==================================================================== #
from __future__ import division
from utils import *
import time
from axis_neuron import left_hand_col_indices,right_hand_col_indices
from port import *
import dill
from subprocess import call



class HandSyncExperiment(object):
    def __init__(self,duration,trial_type,parts_ix=None,broadcast_port=5001,fs=30,rotation_angle=0):
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
        fs : int
            Sampling frequency for interpolated velocities.
        rotation_angle : float
            Radians by which subject would have to be rotated about z-axis (pointing up) to face along the x-axis.
        """
        from shutil import rmtree

        self.duration = duration
        self.trialType = trial_type
        self.broadcastPort = broadcast_port
        self.rotAngle = rotation_angle

        # Clear current directory.
        if len(os.listdir('./'))>0:
            affirm='x'
            while not affirm in 'yn':
                affirm=raw_input("Directory is not empty. Delete files? y/[n]")
            if affirm=='y':
                for f in os.listdir('./'):
                    try:
                        os.remove(f)
                    except OSError:
                        rmtree(f)
            else:
                raise Exception("There are files in current directory.")

    def _load_avatar(self):
        """
        This loads the correct avatar for comparison of performance. The handedness of the subject is
        read in from left_or_right.txt.

        Parameters
        ----------

        Returns
        -------
        avatar : dict
            Dictionary of avatar interpolation splines.
        """
        from pipeline import extract_motionbuilder_model2

        handedness = open('%s/%s'%(DATADR,'left_or_right')).readline().rstrip()
        
        # NOTE: This relies on the fact that these experiments took data from these avatar trials,
        # but you should fetch just the avatar data from the original recordings.
        if handedness=='left':
            v = extract_motionbuilder_model2('avatar',0,'Right',return_time=False)
        elif handedness=='right':
            v = extract_motionbuilder_model2('avatar',0,'Left',return_time=False)
        else:
            print handedness
            raise Exception

        return v

    def _load_avatar(self,return_subject=False):
        """
        This loads the correct avatar for comparison of performance. The handedness of the subject is read in
        from left_or_right.txt.

        Parameters
        ----------
        return_subject : bool,False

        Returns
        -------
        avatar : dict
            Dictionary of avatar interpolation splines.
        """
        from data_access import subject_settings_v3 as subject_settings
        from data_access import VRTrial3_1 as VRTrial
        handedness = open('%s/%s'%(DATADR,'left_or_right')).readline().rstrip()
        
        # NOTE: This relies on the fact that these experiments took data from these avatar trials, but you
        # should fetch just the avatar data from the original recordings.
        if handedness=='left':
            person,modelhandedness,rotation,dr = subject_settings(0)
        elif handedness=='right':
            person,modelhandedness,rotation,dr = subject_settings(2)
        else:
            print handedness
            raise Exception

        trial = VRTrial(person,modelhandedness,rotation,dr)
        avatar = trial.templateTrial
        
        if return_subject:
            subject = trial.subjectTrial
            return avatar,subject
        return avatar

    def load_avatar(self,reverse_time=False,return_subject=False):
        """
        This loads the correct avatar for comparison of performance. The handedness of the subject is read in
        from left_or_right.txt.

        Parameters
        ----------
        return_subject : bool,False

        Returns
        -------
        avatar : dict
            Dictionary of avatar interpolation splines.
        """
        from pipeline import extract_motionbuilder_model3
        handedness = open('%s/%s'%(DATADR,'left_or_right')).readline().rstrip()
        
        if handedness=='left':
            v,t = extract_motionbuilder_model3('Right',reverse_time=reverse_time)
        elif handedness=='right':
            v,t = extract_motionbuilder_model3('Left',reverse_time=reverse_time)
        else:
            print handedness
            raise Exception

        return v

    def wait_for_start(self,dt=.1):
        """
        Wait til start file is written.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start')):
            time.sleep(dt)

    def wait_for_start_time(self):
        """
        Get the time at which the trial started.

        Returns
        -------
        t0 : datetime
            The time at which the trial was started.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start_time')):
            time.sleep(.5)
        # Give some time for the initial time to be written by UE4.
        time.sleep(.5)
        with open('%s/%s'%(DATADR,'start_time'),'r') as f:
            t0 = datetime.strptime(f.readline(),'%Y-%m-%dT%H:%M:%S.%f')
        return t0
   
    def wait_for_start_gpr(self):
        """
        Once start_gpr has been written, erase self.subVBroadcast's memory of the history of
        velocity.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start_gpr')):
            time.sleep(.5)

    def read_pause(self):
        readFile = False
        while not readFile:
            try:
                with open('%s/%s'%(DATADR,'pause_time')) as f:
                    self.pause.append( datetime.strptime(f.readline(),'%Y-%m-%dT%H:%M:%S.%f') )
                readFile = True
            except IOError:
                time.sleep(.02)

    def read_this_setting(self):
        """
        Read in window settings and trial start and end times from this_setting file. Times are appended to
        self.trialStartTimes and self.trialEndTimes.

        Returns
        -------
        thisDuration : float
        thisFraction : float
        """
        # Try to open and read. Sometimes there is a delay in accessibility because
        # the file is being written.
        success = False 
        while not success:
            try:
                with open('%s/%s'%(DATADR,'this_setting')) as f:
                    L = f.readline().split(',')
                    thisDuration,thisFraction = (float(i) for i in L[:2])
                    # start start times with the first trial after full visibility
                    self.trialStartTimes.append( datetime.strptime(L[2],'%Y-%m-%dT%H:%M:%S.%f') )
                    self.trialEndTimes.append( datetime.strptime(L[3],'%Y-%m-%dT%H:%M:%S.%f') )
                success = True
            except IOError:
                pass
        return thisDuration,thisFraction
    
    def delete_file(self,fname,max_wait_time=1,dt=.02):
        """
        Try to delete file in DATADR. Return False if deletion is unsuccessful in given time frame.

        Parameters
        ----------
        fname : str
            Just the file name assuming that it is in DATADR.
        max_wait_time : float
        dt : float
            Time to wait in each iteration of while loop.
        """
        notDeleted = True
        t0 = datetime.now()
        while notDeleted:
            try:
                os.remove('%s/%s'%(DATADR,fname))
                notDeleted = False
            except OSError:
                if (datetime.now()-t0).total_seconds()>max_wait_time:
                    print "Failed to delete %s."%fname
                    return False
                time.sleep(dt)
        print "%s deleted."%fname
        return True

    def run_cal(self,verbose=False,min_v=0.3,pause_before_run=0.):
        """
        Run calibration recording. Have subjects stand straight facing direction of motion. Then, have them
        jerk both hands forward in parallel lines. Then the respective angles for rotating the vector to be
        along the x-axis will be calculated for the left and right hands and put into self.rotAngle.
                
        Parameters
        ----------
        verbose : bool,False
        min_v : float,0.3
            Subject must be moving at least this fast (m/s) along the xy-plane for the calibration to
            register.
        pause_before_run: float,0.
            Number of seconds to wait before running. Used for debugging.
        """
        from numpy.linalg import norm
        counter = 0
        calSuccess = False
        fname = 'an_port_cal.txt'

        while not calSuccess:
            while os.path.isfile(fname):
                fname = 'an_port_cal_%s.txt'%(str(counter).zfill(2))
                counter+=1
            
            raw_input("Press Enter to calibrate...")

            # Setup thread for recording port data.
            recordThread = threading.Thread(target=record_AN_port,
                                            args=(fname,7013),
                                            kwargs={'start_file':'start_cal','stop_file':'stop_cal'})
            time.sleep(pause_before_run)

            print "Running calibration."
            recordThread.start()

            # Run calibration for a few seconds to give people a chance to move their hands.
            with open('start_cal','w') as f:
                f.write('')
            time.sleep(5)
            with open('stop_cal','w') as f:
                f.write('')
            
            time.sleep(.5)

            # Delete signal files.
            print "Done with calibration."
            self.delete_file('start_cal')
            self.delete_file('stop_cal')
            recordThread.join()
            time.sleep(2)

            # Load the data and find which direction the user is facing. Extract from that, the
            # rotation angle needed about the z-axis (pointing up out of the ground) to make the person face the
            # x-axis.
            df = load_AN_port(fname,time_as_dt=False)

            # Get xy vector.
            vright = df.iloc[:,right_hand_col_indices()].values[:,:2]
            vleft = df.iloc[:,left_hand_col_indices()].values[:,:2]
            vright[:,1] *= -1
            vleft[:,1] *= -1
            sright = np.linalg.norm(vright,axis=1)
            sleft = np.linalg.norm(vleft,axis=1)
            
            try:
                # Extract 80 percentile of speed for analysis (as long as it is at least min_v).
                ix = (sright>=np.percentile(sright,80)) & (sright>min_v)
                angleRight = extract_rot_angle(vright[ix])
                ix = (sleft>=np.percentile(sleft,80)) & (sleft>min_v)
                angleLeft = extract_rot_angle(vleft[ix])

                calSuccess = True
            except AssertionError:
                print "Retry calibration."

        self.rotAngle = [-angleLeft,-angleRight]
        print "Rotation angle to center left hand about x-axis is %1.1f degrees."%(
                self.rotAngle[0]*180/np.pi)
        print "Rotation angle to center right hand about x-axis is %1.1f degrees."%(
                self.rotAngle[1]*180/np.pi)

    def run_lf(self,trial_duration):
        """
        Run Leader-Follower experiment.

        Parameters
        ----------
        trial_duration : float
            Duration in seconds for which to record data.
        """
        suffix = 0
        while os.path.isfile('an_port_%s.txt'%str(suffix).zfill(2)):
            suffix += 1

        recordThread = threading.Thread(target=record_AN_port,
                                        args=('an_port_%s.txt'%str(suffix).zfill(2),7013),
                                        kwargs={'start_file':'start_lf','stop_file':'end_lf'})
        with open('start_lf','w') as f:
            f.write('')
        recordThread.start()

        time.sleep(trial_duration)

        with open('end_lf','w') as f:
            f.write('')
        recordThread.join()
        self.delete_file('start_lf')
        self.delete_file('end_lf')

    def run_vr(self,
               update_delay=.3,
               initial_window_duration=1.0,initial_vis_fraction=0.5,
               min_window_duration=.5,max_window_duration=2,
               min_vis_fraction=.1,max_vis_fraction=1.,
               gpr_mean_prior=np.log(.44/.56),
               reverse_time=False,
               verbose=False):
        """
        Run realtime analysis for experiment.
                
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
        
        Notes
        -----
        Code waits til when start_time becomes available to read in avatar start time.
        
        The threads that are running:
        0. reader thread to read velocities from Axis Neuron UDP port rebroadcast 7011.
        1. updateBroadcastThread: assess subject's performance relative to the avatar and
            update performance value
        2. broadcastThread: broadcast subject's performance to port 5001
        3. recordThread: record AN output from UDP rebroadcast @ 7013
        
        Thread communication happens through members that are updated using thread locks.
        
        In while loop, run GPR prediction step and write the next window duration and visible
        fraction to file. Waiting for run_gpr and writing to next_setting.

        When end is written, experiment ends.

        NOTES:
        """
        from data_access import subject_settings_v3
        from data_access import VRTrial3_1 as VRTrial
        from coherence import GPR,DTWPerformance
        self.wait_for_start()
        
        # Setup routines for calculating coherence.
        gprmodel = GPR(mean_performance=gpr_mean_prior,
                       tmin=min_window_duration,tmax=max_window_duration,
                       fmin=min_vis_fraction,fmax=max_vis_fraction)
        realTimePerfEval = DTWPerformance()
        gprPerfEval = DTWPerformance()

        nextDuration = np.around(initial_window_duration,1)
        nextFraction = np.around(initial_vis_fraction,1)
        assert min_window_duration<=nextDuration<=max_window_duration
        assert min_vis_fraction<=nextFraction<=max_vis_fraction

        with open('%s/next_setting'%DATADR,'w') as f:
            f.write('%1.1f,%1.1f'%(nextDuration,nextFraction))
        with open('%s/this_setting'%DATADR,'w') as f:
            f.write('%1.1f,%1.1f,%s,%s'%(0,0,datetime.now().isoformat(),datetime.now().isoformat()))
        with open('%s/left_or_right'%DATADR,'r') as f:
            handedness = f.read().rstrip()  # of subject
        if handedness=='left':
            self.subPartsIx = left_hand_col_indices(False)
            self.avPartsIx = right_hand_col_indices(False)
            rotAngle = self.rotAngle[0]
        else:
            self.subPartsIx = right_hand_col_indices(False)
            self.avPartsIx = left_hand_col_indices(False)
            rotAngle = self.rotAngle[1]
        avatar = self.load_avatar(reverse_time)  # avatar for comparing velocities
        windowsInIndexUnits = int(30*self.duration)
        performance = []  # history of performance
        self.pause = []  # times when game was paused
        self.unpause = []  # times when game was resumed
        self.trialStartTimes = [] # times trials (excluding very first fully visible trial) were started
        self.trialEndTimes = [] # times trials end (including very first fully visible trial) were started
        pauseEvent = threading.Event()
        pauseEvent.set()
        self.endEvent = threading.Event()  # Event to be set when end file is written
        
        # Open port for communication with UE4 engine. This will send the current coherence value to
        # UE4.
        self.broadcast = DataBroadcaster(self.broadcastPort)
        self.broadcast.update_payload('-1.0')
        broadcastThread = threading.Thread(target=self.broadcast.broadcast,
                   kwargs={'pause':.2,'verbose':True if verbose=='detailed' else False})
        broadcastThread.start()

        # Setup thread for recording port data.
        recordThread = threading.Thread(target=record_AN_port,args=('an_port.txt',7013))

        # Set up thread for updating value of streaming broadcast of coherence.
        # This relies on reader to fetch data which is declared later.
        def update_broadcaster(reader,stopEvent):
            try:
                while not stopEvent.is_set():
                    pauseEvent.wait()
                    v,t,tAsDate = reader.copy_recent()
                    
                    if len(v)>=(windowsInIndexUnits):
                        # Put into standard coordinate system (as in paper). Account for reflection symmetry.
                        v[:,1:] *= -1
                        v[:,:2] = rotate_xy(v[:,:2],rotAngle)

                        tAsDate,_ = remove_pause_intervals(tAsDate.tolist(),zip(self.pause,self.unpause))
                        avv = fetch_matching_avatar_vel(avatar,np.array(tAsDate),t0)
                        
                        # Calculate performance metric.
                        performance.append( realTimePerfEval.raw(v[:,1:],avv[:,1:],dt=1/30) )

                        # Update performance.
                        self.broadcast.update_payload('%1.2f'%performance[-1])
                        if verbose=='detailed':
                            print "new coherence is %s"%self.broadcast._payload
                    time.sleep(0.2)
            finally:
                print "updateBroadcastThread stopped"
        self.updateBroadcastEvent = threading.Event()

        # Define function that will be run in GPR thread. 
        def run_gpr_update(reader,gprmodel):
            """
            Run GPR updater when run_gpr appears. Read in current window setting from this_setting and write
            out next window setting to next_setting.
            """
            while not self.endEvent.is_set():
                pauseEvent.wait()

                # Run GPR for the next windows setting.
                if os.path.isfile('%s/%s'%(DATADR,'run_gpr')):
                    print "Running GPR on this trial..."
                    v,t,tdateHistory = reader.copy_history()
                    # Put output from Axis Neuron into comparable coordinate system accounting for reflection
                    # symmetry.
                    v[:,1:] *= -1
                    v[:,:2] = rotate_xy(v[:,:2],rotAngle)

                    tdateHistory,_ = remove_pause_intervals(tdateHistory.tolist(),
                                                            zip(self.pause,self.unpause))
                    avv = fetch_matching_avatar_vel(avatar,np.array(tdateHistory),t0)

                    # Try to open and read. Sometimes there is a delay in accessibility because
                    # the file is being written.
                    thisDuration,thisFraction = self.read_this_setting()
                    
                    # Get subject performance ignoring the first few seconds of performance.
                    perf = gprPerfEval.time_average( avv[75:,1:],v[75:,1:],dt=1/30 )
                    
                    # Update GPR. For initial full visibility trial, update values for all values of fraction.
                    if thisDuration==0:
                        gprmodel.update( ilogistic(perf),0.,1. )
                        nextDuration,nextFraction = gprmodel.max_uncertainty()
                    else:
                        gprmodel.update( ilogistic(perf),thisDuration,thisFraction )
                        nextDuration,nextFraction = gprmodel.max_uncertainty()
                    if verbose:
                        #print call("ls --time-style='+%d-%m-%Y %H:%M:%S' -l this_setting",shell=True)
                        print "thisDuration: %1.1f\tthisFraction: %1.1f"%(thisDuration,thisFraction)
                        print "nextDuration: %1.1f\tnextFraction: %1.1f"%(nextDuration,nextFraction)
                    open('%s/next_setting'%DATADR,'w').write('%1.1f,%1.1f'%(nextDuration,
                                                                            nextFraction))
                    
                    # Delete signal file.
                    self.delete_file('run_gpr')

                    # Refresh history.
                    self.broadcast.update_payload('-1.0')
                    reader.refresh()
                    while reader.len_history()<(self.duration*30):
                        if verbose:
                            print "Waiting to collect more data...(%d)"%reader.len_history()
                        time.sleep(.5)
                    dill.dump({'gprmodel':gprmodel,'performance':performance,
                               'pause':self.pause,'unpause':self.unpause,
                               'trialStartTimes':self.trialStartTimes,
                               'trialEndTimes':self.trialEndTimes},
                              open('%s/%s'%(DATADR,'gpr.p'),'wb'),-1)

                time.sleep(.05)


        # Wait til start_time has been written to start experiment.
        t0 = self.wait_for_start_time()

        if verbose:
            print "Starting threads."
        recordThread.start()
        with ANReader(self.duration,self.subPartsIx,
                      port=7011,
                      verbose=True if verbose=='detailed' else False,
                      port_buffer_size=8192,
                      recent_buffer_size=(self.duration+1)*30) as reader:
            
            updateBroadcastThread = threading.Thread(target=update_broadcaster,
                                                     args=(reader,self.updateBroadcastEvent))

            while reader.len_history()<windowsInIndexUnits:
                if verbose:
                    print "Waiting to collect more data...(%d)"%reader.len_history()
                self.broadcast.update_payload('-1.0')
                time.sleep(1)
            updateBroadcastThread.start()
           
            # Start GPR thread.
            gprThread = threading.Thread(target=run_gpr_update,args=(reader,gprmodel))
            gprThread.start()

            while not os.path.isfile('%s/%s'%(DATADR,'end')):
                # If UE4 has been paused
                if os.path.isfile('%s/%s'%(DATADR,'pause_time')):
                    pauseEvent.clear()
                    if verbose: print "Paused."
                    self.read_pause()
                    self.delete_file('pause_time')
                    while not os.path.isfile('%s/%s'%(DATADR,'unpause_time')):
                        time.sleep(.01)
                    
                    # Try to open and read unpause_time. Sometimes there is a delay in accessibility because
                    # the file is being written.
                    success = False 
                    while not success:
                        try:
                            with open('%s/%s'%(DATADR,'unpause_time')) as f:
                                self.unpause.append( datetime.strptime(f.readline(),'%Y-%m-%dT%H:%M:%S.%f') )
                            success = True
                            pauseEvent.set()
                            if verbose: print "Unpaused."
                        except IOError:
                            pass
                    self.delete_file('unpause_time')
                    
                    reader.refresh()
                    while reader.len_history()<(self.duration*30):
                        if verbose:
                            print "Waiting to collect more data...(%d)"%reader.len_history()
                        time.sleep(.5)
                
                time.sleep(.1)
            
        print "Ending threads..."
        self.stop()
        updateBroadcastThread.join()
        broadcastThread.join()
        recordThread.join()
        gprThread.join()

        with open('%s/%s'%(DATADR,'end_port_read'),'w') as f:
            f.write('')

        # Read in last trial setting.
        self.read_this_setting() 
        
        print "Saving GPR."
        dill.dump({'gprmodel':gprmodel,'performance':performance,
                   'pause':self.pause,'unpause':self.unpause,
                   'trialStartTimes':self.trialStartTimes,
                   'trialEndTimes':self.trialEndTimes,
                   'rotAngle':rotAngle},
                  open('%s/%s'%(DATADR,'gpr.p'),'wb'),-1)
        
        # Move all files into the left or right directory given by which hand the subject was using.
        if not os.path.isdir(handedness):
            os.mkdir(handedness)
        for f in ['left_or_right','end','initial_trial_done','run_gpr','start','start_time','this_setting',
                  'next_setting','an_port.txt','end_port_read','gpr.p','an_port_cal.txt']:
            if os.path.isfile(f):
                os.rename(f,'%s/%s'%(handedness,f))

    def stop(self):
        """Stop all thread that could be running. This does not wait for threads to stop."""
        self.updateBroadcastEvent.set()
        self.broadcast.stopEvent.set()
        self.endEvent.set()
        return
# end HandSyncExperiment



def fetch_matching_avatar_vel(avatar,t,t0=None,verbose=False):
    """
    Get the stretch of avatar velocities that aligns with the velocity data of the subject. 

    Parameters
    ----------
    avatar : dict
        This would be the templateTrial loaded in VRTrial.
    t : array of floats or datetime objects
        Stretch of time to return data from. If t0 is specified, this needs to be datetime objects.
    t0 : datetime,None
    verbose : bool,False

    Returns
    -------
    v : ndarray
        (n_time,3). Avatar's velocity that matches given time stamps.
    """
    if not t0 is None:
        # Transform dt to time in seconds.
        t = np.array([i.total_seconds() for i in t-t0])
        assert (t>=0).all()
    if verbose:
        print "Getting avatar times between %1.1fs and %1.1fs."%(t[0],t[-1])

    # Return part of avatar's trajectory that agrees with the stipulated time bounds.
    return avatar(t)

def remove_pause_intervals(t,pause_intervals):
    """
    Given a list of time points where data was taken and a list of tuples where the data take was paused,
    return the times at which the data would've been taken if there had been no pause having removed all data
    points that were taken during the specified pause intervals.

    Parameters
    ----------
    t : list
        datetime.datetime objects of when data was recorded. These should be ordered in time.
    pause_intervals : list of tuples
        Each tuple should be (start,end).

    Returns
    -------
    tDate : list of datetime.datetime objects
    t : ndarray
        Time is seconds starting from tDate[0]
    """
    t = t[:]
    pause_intervals = pause_intervals[:]

    for dtix,(t0,t1) in enumerate(pause_intervals):
        assert t0<t1
        dt = t1-t0
        counter = 0
        t_ = t[counter]
        while t_<t0 and counter<len(t):
            t_ = t[counter]
            counter += 1

        # Remove all data points within the pause interval.
        if counter>0:
            while t[counter-1] < t1:
                t.pop(counter-1)
        
        # If none of the pause intervals overlap with the given data.
        if counter<len(t):
            for counter in xrange(counter-1,len(t)):
                t[counter] -= dt
            for dtix in xrange(dtix+1,len(pause_intervals)):
                pause_intervals[dtix] = (pause_intervals[dtix][0]-dt,pause_intervals[dtix][1]-dt)
    return t,np.concatenate([[0],np.cumsum([i.total_seconds() for i in np.diff(t)])])

def ilogistic(x):
    return -np.log(1/x-1)

def logistic(x):
    return 1/(1+np.exp(-x))

def extract_rot_angle(v,noise_threshold=.4,min_points=10):
    """
    Take average normalized vector and use that to calculate rotation angle of vector.  There can be a set of
    velocities for forward and then backwards movement that are both used to get a better approximation of the
    direction of motion  Rotation angle is along direction of initial movement.

    This needs to be negated to get the angle that we need to rotate about the z-axis to get the vector to
    point along the x-axis.
    
    Parameters
    ----------
    v : ndarray
        2d velocity measurements. (n_samples,2)
    noise_threshold : float,0.4
        Allowed noise in fluctuation of angle amongst given time points. If it is too 
        large, an error is thrown.

    Returns
    -------
    rotAngle : float
        Angle of initial velocity.
    """
    from misc.angle import mod_angle
    assert len(v)>min_points
    vnorm = np.linalg.norm(v,axis=1)[:,None]
    assert (vnorm>0).all(),"Zero velocities not allowed."
    v = v/vnorm
    
    # Orient velocities such that velocity vectors when hands are moving back are facing 
    # in the same direction as when moving forwards.
    angle = np.arctan2(v[:,1],v[:,0])
    ix = np.abs(mod_angle(angle[0]-angle)) > (.5*np.pi)
    assert ix.any(), "Failed to get both forward and backward directions."
    assert np.diff(ix).sum()==1, "All vectors should point in same direction before and after switch in direction."
    
    # Check noise by looking at spread of angle.
    v[ix] = rotate_xy(v[ix],np.pi)
    angle = np.arctan2(v[:,1],v[:,0])
    assert mod_angle(angle[0]-angle[1:]).std()<noise_threshold, "Angle measurements are noisy."
    
    # Calculate final value by averaging across vectors.
    v = v.mean(0)
    v /= np.linalg.norm(v)
    
    return np.arctan2(v[1],v[0])
