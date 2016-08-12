# 2016-08-11
from __future__ import division
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
    from itertools import chain
    from pyparsing import nestedExpr
    
    # Parse skeleton.
    # Find the line where data starts and get skeleton tree lines.
    s = ''
    lineix = 0
    with open(fname) as f:
        f.readline()
        f.readline()
        ln = f.readline()
        lineix += 3
        while not 'MOTION' in ln:
            s += ln
            ln = f.readline()
            lineix += 1
        
        # Read in the frame rate.
        while 'Frame Time' not in ln:
            ln = f.readline()
            lineix += 1
        dt = float( ln.split(' ')[-1][:-2] )
    
    s = nestedExpr('{','}').parseString(s).asList()
    nodes = []

    def parse(parent,thisNode,skeleton):
        """
        Keep track of traversed nodes in nodes list.

        Params:
        -------
        parent (str)
        skeleton (list)
            As returned by pyparsing
        """
        children = []
        for i,ln in enumerate(skeleton):
            if (not type(ln) is list) and 'JOINT' in ln:
                children.append(skeleton[i+1])
            elif type(ln) is list:
                if len(children)>0:
                    parse(thisNode,children[-1],ln)
        nodes.append( Node(thisNode,parents=[parent],children=children) )

    parse('','Hips',s[0])
    bodyParts = [n.name for n in nodes]
    skeleton = Tree(nodes) 
   
    # Parse motion.
    df = pd.read_csv(fname,skiprows=lineix+2,delimiter=' ',header=None)
    df = df.iloc[:,:-1]  # remove bad last col

    if includeDisplacement:
        df.columns = pd.MultiIndex.from_arrays([list(chain.from_iterable([[b]*6 for b in bodyParts])),
                                            ['xx','yy','zz','y','x','z']*len(bodyParts)])
    else:
        df.columns = pd.MultiIndex.from_arrays([[bodyParts[0]]*6 + 
                                                 list(chain.from_iterable([[b]*3 for b in bodyParts[1:]])),
                                            ['xx','yy','zz']+['y','x','z']*len(bodyParts)])
   

    # Filtering.
    if removeBlank:
        # Only keep entries that change at all.
        df = df.iloc[:,np.diff(df,axis=0).sum(0)!=0] 
    return df,dt,skeleton


class Node(object):
    def __init__(self,name=None,parents=[],children=[]):
        self.name = name
        self.parents = parents
        self.children = children

    def add_child(self,child):
        self.children.append(child)

class Tree(object):
    def __init__(self,nodes):
        self._nodes = nodes
        self.nodes = [n.name for n in nodes]
        names = [n.name for n in nodes]
        if len(np.unique(names))<len(names):
            raise Exception("Nodes have duplicate names.")

        self.adjacency = np.zeros((len(nodes),len(nodes)))
        for i,n in enumerate(nodes):
            for c in n.children:
                try:
                    self.adjacency[i,names.index(c)] = 1
                # automatically insert missing nodes (these should all be dangling)
                except ValueError:
                    self.adjacency = np.pad( self.adjacency, ((0,1),(0,1)), mode='constant', constant_values=0)
                    self._nodes.append( Node(c) )
                    names.append(c)

                    self.adjacency[i,names.index(c)] = 1
        
    def print_tree(self):
        print self.adjacency

