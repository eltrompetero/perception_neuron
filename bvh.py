# 2016-08-11

import pandas as pd
import numpy as np

def load(fname,includeDisplacement=False,removeBlank=True):
    """
    Load data from BVD file.
    2016-08-11

    Params:
    -------
    fname (str)
        Name of file to load
    includeDisplacement (bool=False)
        If displacement data is included for everything including root.
    removeBlank (bool=True)
        Remove entries where nothing changes over the entire recording session. This should mean that there was nothing being recorded in that field.

    Value:
    ------
    df (dataFrame)
    dt (float)
        Frame rate.
    """
    # Find the line where data starts and get skeleton parts.
    from itertools import chain
    bodyParts = []
    with open(fname) as f:
        i = 0
        ln = f.readline()
        while not 'ROOT' in ln:
            ln = f.readline()
            i += 1
        bodyParts.append( ''.join(a for a in ln.split(' ')[1] if a.isalnum()) )
        while not 'Frames' in ln:
            if 'JOINT' in ln:
                ix = ln.find('JOINT')
                bodyParts.append( ''.join(a for a in ln[ix:].split(' ')[1] if a.isalnum()) )
            ln = f.readline()
            i += 1

        # Read in the frame rate.
        while 'Frame Time' not in ln:
            ln = f.readline()
        dt = float( ln.split(' ')[-1][:-2] )

    df = pd.read_csv(fname,skiprows=i+2,delimiter=' ',header=None)
    df = df.iloc[:,:-1]  # remove bad last col
    
    if includeDisplacement:
        df.columns = pd.MultiIndex.from_arrays([list(chain.from_iterable([[b]*6 for b in bodyParts])),
                                            ['xx','yy','zz','y','x','z']*len(bodyParts)])
    else:
        df.columns = pd.MultiIndex.from_arrays([[bodyParts[0]]*6 + 
                                                 list(chain.from_iterable([[b]*3 for b in bodyParts[1:]])),
                                            ['xx','yy','zz']+['y','x','z']*len(bodyParts)])
    
    if removeBlank:
        # Only keep entries that change at all.
        df = df.iloc[:,np.diff(df,axis=0).sum(0)!=0] 
    return df,dt

