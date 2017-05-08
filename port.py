# ===================================================================================== #
# Module for reading data from Axis Neuron broadcast port.
# Author: Eddie Lee edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from datetime import datetime,timedelta
import os,time,pause,socket

HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = 7003

# Functions for reading from broadcasting port.
def read_port():
    def incoming(host, port):
        """Open specified port and return file-like object"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.connect((host,port))
        data=sock.recvfrom(32768)
        sock.close()
        return data[0]
    data = incoming(HOST, PORT)
    return datetime.now(),data

def format_port_output(s):
    s=s.split(' ')[1:11]
    f=[float(i) for i in s]
    return f

def _format_port_output(s):
    """For BVH file."""
    ix=s.find('Caeli')
    if ix==-1:
        return ['NaN']*10
    s=s[ix:].split(' ')[1:11]
    f=[float(i) for i in s]
    return f
# end port section

def record_AN_port(fname,recStartTime,recEndTime=None,dt=None):
    """
    Start recording data from Axis Neuron port at some predetermined time and saves it to file name.
    
    Params:
    -------
    fname (str)
    startTime (float or datetime.datetime)
    endTime (datetime.datetime=None)
    dt (float=None)
        If this is given instead of endTime, then record for dt seconds.
    """
    if type(recStartTime) is int or type(recStartTime) is float:
        recStartTime = datetime.now() + timedelta(seconds=recStartTime)
    else:
        assert type(recStartTime) is datetime.datetime, "recStartTime must be seconds or datetime object."
    if recEndTime is None:
        recEndTime = recStartTime + timedelta(seconds=dt)
    
    data = []  # Port output.
    pause.until(recStartTime)
    while datetime.now()<recEndTime:
        data.append(read_port())
    
    with open('test.txt','w') as f:
        f.write('Start time: %s\n'%data[0][0].isoformat())
        f.write('End time: %s\n\n'%data[-1][0].isoformat())
        for d in data:
            f.write('%s %s\n'%(d[0].isoformat(),d[1]))

