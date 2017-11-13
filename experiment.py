# ==================================================================== #
# Class for running synchronization experiments.
# Author: Eddie Lee edl56@cornell.edu
# ==================================================================== #
from __future__ import division
from utils import *
import time
from axis_neuron import left_hand_col_indices,right_hand_col_indices
from port import *

class HandSyncExperiment(object):
    def __init__(self,duration,trial_type,parts_ix=None,broadcast_port=5001,fs=30):
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
        """
        self.duration = duration
        self.trialType = trial_type
        self.broadcastPort = broadcast_port

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

    def load_avatar(self,return_subject=False):
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
        from data_access import VRTrial
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

    def wait_for_start(self):
        """
        Wait til start file is written.
        """
        while not os.path.isfile('%s/%s'%(DATADR,'start')):
            time.sleep(.5)

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
    
    def delete_file(self,fname,max_wait_time=1,dt=.02):
        """
        Try to delete file in DATADR. Return False if deletion is unsuccessful in given time frame.

        Parameters
        ----------
        fname : str
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

    def start(self,
              update_delay=.3,
              initial_window_duration=1.0,initial_vis_fraction=0.5,
              min_window_duration=.6,max_window_duration=2,
              min_vis_fraction=.1,max_vis_fraction=.9,
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
        3. recordThread: record AN output from UDP rebroadcast @ 7010
        
        Thread communication happens through members that are updated using thread locks.
        
        In while loop, run GPR prediction step and write the next window duration and visible
        fraction to file. Waiting for run_gpr and writing to next_setting.

        When end is written, experiment ends.

        NOTE:
        - only calculate coherence along z-axis
        """
        from data_access import subject_settings_v3,VRTrial
        from coherence import GPR,DTWPerformance
        import cPickle as pickle
        self.wait_for_start()
        
        # Setup routines for calculating coherence.
        gprmodel = GPR(tmin=min_window_duration,tmax=max_window_duration,
                       fmin=min_vis_fraction,fmax=max_vis_fraction)
        realTimePerfEval = DTWPerformance()
        gprPerfEval = DTWPerformance()

        nextDuration = np.around(initial_window_duration,1)
        nextFraction = np.around(initial_vis_fraction,1)
        assert min_window_duration<=nextDuration<=max_window_duration
        assert min_vis_fraction<=nextFraction<=max_vis_fraction

        with open('%s/next_setting'%DATADR,'w') as f:
            f.write('%1.1f,%1.1f'%(nextDuration,nextFraction))
        with open('%s/left_or_right'%DATADR,'r') as f:
            handedness = f.read().rstrip()
        if handedness=='left':
            self.subPartsIx = left_hand_col_indices(False)
            self.avPartsIx = right_hand_col_indices(False)
        else:
            self.avPartsIx = left_hand_col_indices(False)
            self.subPartsIx = right_hand_col_indices(False)
        avatar = self.load_avatar()['avatarV']  # avatar for comparing velocities
        windowsInIndexUnits = int(30*self.duration)
        performance = []  # history of performance
        
        # Open port for communication with UE4 engine. This will send the current coherence value to
        # UE4.
        self.broadcast = DataBroadcaster(self.broadcastPort)
        self.broadcast.update_payload('-1.0')
        broadcastThread = threading.Thread(target=self.broadcast.broadcast,
                                           kwargs={'pause':.2,'verbose':verbose})
        broadcastThread.start()

        # Setup thread for recording port data.
        recordThread = threading.Thread(target=record_AN_port,args=('an_port.txt',7010))

        # Set up thread for updating value of streaming broadcast of coherence.
        # This relies on reader to fetch data which is declared later.
        def update_broadcaster(reader,stopEvent):
            try:
                while not stopEvent.is_set():
                    v,t,tAsDate = reader.copy_recent()
                    
                    if len(v)>=(windowsInIndexUnits):
                        # Put into standard coordinate system (as in paper). Account for reflection symmetry.
                        v[:] = v[:,[1,0,2]]
                        v[:,2] *= -1
                        avv = fetch_matching_avatar_vel(avatar,tAsDate,t0)
                        # Template avatar motion has been modified to account for reflection symmetry of left
                        # and right hand motions.
                        avv[:,1] *= -1
                        
                        # Calculate performance metric.
                        performance.append( realTimePerfEval.raw(v,avv,dt=1/30) )

                        # Update performance.
                        self.broadcast.update_payload('%1.2f'%performance[-1])
                        print "new coherence is %s"%self.broadcast._payload
                    time.sleep(0.1)
            finally:
                print "updateBroadcastThread stopped"
        self.updateBroadcastEvent = threading.Event()

        # Wait til start_time has been written to start experiment..
        t0 = self.wait_for_start_time()

        if verbose:
            print "Starting threads."
        recordThread.start()
        print self.subPartsIx
        with ANReader(self.duration,self.subPartsIx,
                      port=7011,
                      verbose=True,
                      port_buffer_size=8192,
                      recent_buffer_size=self.duration*60) as reader:
            
            updateBroadcastThread = threading.Thread(target=update_broadcaster,
                                                     args=(reader,self.updateBroadcastEvent))

            while reader.len_history()<windowsInIndexUnits:
                if verbose:
                    print "Waiting to collect more data...(%d)"%reader.len_history()
                self.broadcast.update_payload('-1.0')
                time.sleep(1)
            #    raise Exception
            updateBroadcastThread.start()
           

            # After initial trial is done, refresh history of GPR.
            while not os.path.isfile('%s/%s'%(DATADR,'initial_trial_done')):
                time.sleep(.2)
            self.broadcast.update_payload('-1.0')
            reader.refresh()
            self.delete_file('initial_trial_done')

            while not os.path.isfile('%s/%s'%(DATADR,'end')):
                # Run GPR for the next windows setting.
                if os.path.isfile('%s/%s'%(DATADR,'run_gpr')):
                    print "Running GPR on this trial..."
                    v,t,tdateHistory = reader.copy_history()
                    # Put into comparable coordinate system accounting for reflection symmetry.
                    v[:] = v[:,[1,0,2]]
                    v[:,2] *= -1

                    avv = fetch_matching_avatar_vel(avatar,tdateHistory,t0)
                    # Template avatar motion has been modified to account for reflection symmetry of left
                    # and right hand motions.
                    avv[:,1] *= -1

                    perf = gprPerfEval.time_average( avv,v,dt=1/30 )
                    nextDuration,nextFraction = gprmodel.update( perf,nextDuration,nextFraction )
                    open('%s/next_setting'%DATADR,'w').write('%1.1f,%1.1f'%(nextDuration,
                                                                            nextFraction))

                    self.delete_file('run_gpr')

                    # Refresh history.
                    self.broadcast.update_payload('-1.0')
                    reader.refresh()
                    while reader.len_history()<(self.duration*60):
                        if verbose:
                            print "Waiting to collect more data...(%d)"%reader.len_history()
                        time.sleep(1)
                    
                time.sleep(update_delay) 
            
        # Always end thread.
        print "Ending threads..."
        self.stop()
        updateBroadcastThread.join()
        broadcastThread.join()
        recordThread.join()

        with open('%s/%s'%(DATADR,'end_port_read'),'w') as f:
            f.write('')

        print "Saving GPR."
        pickle.dump({'gprmodel':gprmodel,'performance':performance,'v':v,'t':t},
                    open('%s/%s'%(DATADR,'temp.p'),'wb'),-1)

    def stop(self):
        """Stop all thread that could be running. This does not wait for threads to stop."""
        self.updateBroadcastEvent.set()
        self.broadcast.stopEvent.set()
        return
# end HandSyncExperiment

def fetch_matching_avatar_vel(avatar,t,t0=None,verbose=False):
    """
    Get the stretch of avatar velocities that aligns with the velocity data of the subject. 

    Parameters
    ----------
    avatar : dict
        This would be the templateTrial loaded in VRTrial.
    part : str
        Choose from 'avatar','avatar0','hand','hand0'.
    t : array of floats or datetime objects
        Stretch of time to return data from. If t0 is specified, this needs to be datetime objects.
    t0 : datetime,None
    verbose : bool,False

    Returns
    -------
    v : ndarray
        Avatar's velocity that matches given time stamps.
    """
    if not t0 is None:
        # Transform dt to time in seconds.
        t = np.array([i.total_seconds() for i in t-t0])
    if verbose:
        print "Getting avatar times between %1.1fs and %1.1fs."%(t[0],t[-1])

    # Return part of avatar's trajectory that agrees with the stipulated time bounds.
    return avatar(t)


