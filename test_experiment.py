from __future__ import division
from experiment import *
from coherence import DTWPerformance


def test_extract_rot_angle():
    """
    Test basic error checking in extract_rot_angle and that it can identify a simple forwards/backwards set of
    motion.
    """
    v = np.zeros((4,2))
    try:
        angle = extract_rot_angle(v,min_points=0)
    except AssertionError,err:
        assert err.args[0]=="Zero velocities not allowed."
    
    v[:,1] = 1.
    try:
        angle = extract_rot_angle(v,min_points=0)
    except AssertionError,err:
        assert err.args[0]=="Failed to get both forward and backward directions."

    # Forwards-backwards motion.
    v[:,1] = 0.
    v[:2,0] = -1.1
    v[2:,0] = 1.2
    angle = extract_rot_angle(v,min_points=0)
    assert np.isclose(angle,np.pi)

    # Forwards-backwards motion.
    v[:,0] = 0.
    v[:2,1] = -.9
    v[2:,1] = .8
    angle = extract_rot_angle(v,min_points=0)
    assert np.isclose(angle,-np.pi/2)

    # Forwards-backwards motion with noise.
    v[:2,1] += (np.random.rand(2)*2-1)/10
    v[2:,1] += (np.random.rand(2)*2-1)/10
    angle = extract_rot_angle(v,min_points=0)
    assert np.isclose(angle,-np.pi/2,atol=.1)

def test_logistic():
    """Test logistic and inverse logistic functions."""
    r=np.random.normal(size=20)
    assert np.isclose( ilogistic(logistic(r)),r ).all()

def test_remove_pause_intervals():
    remove_pause_intervals([datetime.now()],[])

def test_update_broadcast():
    """Generate fake velocity data set. Check that performance evaluation is able to match up the avatar and
    subject velocities and do the evaluation. Since the trajectories are the same, performance should be 1
    within numerical precision.
    """
    # Generate trajecotry.
    t=np.arange(100)/30
    r=interp1d(t,np.vstack((np.zeros((10,3)),
                 np.random.rand(10,3),
                 np.zeros((80,3)))),axis=0)
    t0=datetime.now()

    class VirtualReader(object):
        def copy_recent(self):
            # Account for coordinate system transform.
            r_=r(np.arange(10,20)/30)
            r_[:,1:]*=-1
            t=t0
            return ( r_,
                    np.linspace(0,1,10),
                    np.array([t0+timedelta(seconds=(i+10)/30) for i in range(10)]) )
        
    class VirtualBroadcaster(object):
        def __init__(self):
            self._payload=0.
            
        def update_payload(self,x):
            return
        
    reader=VirtualReader()
    pauseEvent=threading.Event()
    pauseEvent.set()
    stopEvent=threading.Event()
    perfEval=DTWPerformance()
    broadcast=VirtualBroadcaster()
    rotAngle=0.
    avatar=lambda x: r(x)
    performance=[]
    
    # Run in experiment.
    experiment=HandSyncExperiment(2,'avatar',check_directory=False)
    updateBroadcaster=experiment.define_update_broadcaster(reader,stopEvent,pauseEvent,
                                                           5,perfEval,broadcast,
                                                           rotAngle,avatar,t0)
    testThread=threading.Thread(target=updateBroadcaster,args=(performance,))

    testThread.start()
    time.sleep(1)

    stopEvent.set()
    assert (np.array(performance)>.99).all()
