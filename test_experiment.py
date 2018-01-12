from __future__ import division
from experiment import *

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

