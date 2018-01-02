from __future__ import division
from experiment import *

def test_extract_rot_angle():
    v = np.zeros((4,2))
    try:
        angle = extract_rot_angle(v,min_points=0)
    except AssertionError,err:
        assert err.args[0]=="Zero velocities are present."
    
    v[:,0] = 1.
    try:
        angle = extract_rot_angle(v,min_points=0)
    except AssertionError,err:
        assert err.args[0]=="Failed to get both forward and backward directions."

    v[:2,0] = -1.
    v[2:,0] = 1.

    angle = extract_rot_angle(v,min_points=0)
    assert angle==np.pi
