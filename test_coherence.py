from .coherence import *
from itertools import combinations
from scipy.spatial.distance import squareform
gpr = GPR()

def test_scale_gauss():
    assert np.isclose( gpr._scale_erf(0,0,1), 0.5 )
    assert np.isclose( gpr._scale_erf(1,1,1), 0.5 )
    assert np.isclose( gpr._scale_erf(2,1,1), 0.841344746,atol=1e-6 )

def test_kernel(n_tries=10,n_points=50):
    """
    Sample from space to check if we can be relatively confident that kernel is positive definite.
    """
    kernel = gpr.kernel

    # Try on n_tries random matrices.
    for i in range(10):
        randomPts = np.vstack((np.random.uniform(gpr.tmin,gpr.tmax,size=n_points),
                               np.random.uniform(gpr.fmin,gpr.fmax,size=n_points))).T
        c = np.zeros(n_points*(n_points-1)//2)

        for counter,(ii,jj) in enumerate(combinations(list(range(n_points)),2)):
            c[counter] = kernel(randomPts[ii],randomPts[jj])
        c = squareform(c)
        for ii in range(n_points):
            c[ii,ii] = kernel(randomPts[ii],randomPts[ii])

        assert (np.linalg.eigvals(c)>-1e-14).all()


