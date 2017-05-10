# ===================================================================================== #
# Module for reading data from Axis Neuron broadcast port.
# Author: Eddie Lee edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from datetime import datetime,timedelta
import os,time,pause,socket
import pandas as pd
from load import calc_file_headers

HOST = '127.0.0.1'   # use '' to expose to all networks
PORT = 7003  # Calculation data.

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
    portOut = [datetime.now()]*2
    pause.until(recStartTime)
    while portOut[0]<recEndTime:
        portOut = read_port()
        data.append(portOut)
    
    headers = list(calc_file_headers())
    headers[-1] = ''.join(headers[-1].split())  # Remove space in last column header.
    with open(fname,'w') as f:
        f.write('Start time: %s\n'%data[0][0].isoformat())
        f.write('End time: %s\n\n'%data[-1][0].isoformat())
        f.write('Timestamp '+' '.join(headers)+'\n')
        for d in data:
            t = d[0].isoformat()
            if len(t)<10:
                t = 'NaN'
            f.write('%s %s'%(t,d[1]))

def load_AN_port(fname,time_as_dt=True,n_avatars=1):
    """
    With data from a single individual at this moment.
    
    Params:
    -------
    fname (str)
    time_as_dt (bool=True)
    """
    #with open(fname,'r') as f:
    #    startTime = datetime.strptime( f.readline().split(' ')[-1] )
    #    stopTime = datetime.strptime( f.readline().split(' ')[-1] )
    
    df = pd.read_csv(fname,delimiter=' ',skiprows=3)
    df.ix[:,0] = df.ix[:,0].apply(lambda t: datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f'))

    if time_as_dt:
        # Convert time stamp into time differences in seconds. This means we have to remove the first data
        # point.
        dt = np.diff(df.ix[:,0]).astype(int)/1e9
        df = df.ix[1:,:]
        df['Timestamp'] = df['Timestamp'].apply(pd.to_numeric,errors='coerce')
        df['Timestamp'] = dt
    return df

