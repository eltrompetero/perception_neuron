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
            #if '\r' in d[1] or '\n' in d[1]:
            #    raise Exception
            f.write('%s %s\n'%(t,d[1].rstrip()))

def _fix_problem_dates(f,fname):
    """
    Insert missing datetime or missing microseconds at beginning of line. Put in 1900-01-01T00:00:00.000 if
    missing date completely.
    """
    with open('/tmp/temp.txt','w') as fout:
        # skip header lines
        for i in xrange(5):
            fout.write(f.readline())

        for ln in f:
            try:
                d = datetime.strptime(ln[:26], '%Y-%m-%dT%H:%M:%S.%f')
            except ValueError:
                if len(ln[:26].split()[0])==19:
                    print "Inserting microseconds."
                    ln = ln.split()
                    ln[0] += '.000000'
                    ln = ' '.join(ln)+'\n'
                else:
                    print "Adding in date."
                    ln = '1900-01-01T00:00:00.000000 '+ln
                # Sometimes, a single port broadcost seems to overlap with another.
                if len(ln.split())>948:
                    ln = ' '.join(ln.split()[:948])+'\n'
            fout.write(ln) 
    os.rename('/tmp/temp.txt',fname)

def load_AN_port(fname,dr='',time_as_dt=True,n_avatars=1,fix_file=True):
    """
    With data from a single individual at this moment.
    
    Params:
    -------
    fname (str)
    dr (str='')
    time_as_dt (bool=True)
    fix_file (bool=True)
        Parse input file and fix any problems with dates.
    """
    if len(dr)>0:
        fname = '%s/%s'%(dr,fname)
    
    if fix_file:
        # Insert missing times for the file to be read in.
        with open(fname,'r') as f:
            _fix_problem_dates(f,fname) 

    # Read in start and stop times at beginning of file.
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

