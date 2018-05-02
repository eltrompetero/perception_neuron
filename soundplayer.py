# =============================================================================================== #
# Module for writing mocap files to audio. 
# Author : Saerom Choi
# =============================================================================================== #
import os, wave, struct, math, sys, numpy as np, matplotlib.pyplot as plt
from perceptionneuron.pipeline import extract_motionbuilder_model3_3,extract_motionbuilder_Eddie_Grid_Model_2
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit
import numpy.fft as fft
import pyaudio

WAV_SAMPLERATE = 44100.0



def load_sound(filename):
    wav = wave.open(filename,'r')
    sig = wav.readframes('-1')
    sif = np.fromstring(sig,'int16')
    return sif

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

    


class SoundPlayer():
    """Class for converting motion into a frequency varying sound file."""
    def __init__(self, directory, model_name, direction, reverse_time):
        self.models = {'3_3': extract_motionbuilder_model3_3,
                       'Eddie_Grid_Model_2': extract_motionbuilder_Eddie_Grid_Model_2 }
        self.model = self.models[model_name]
        self.function_dict = {'exp': (np.exp, 0.75),
                              'log': (np.log,0.5),
                              'sig': (expit,0.5)}
        self.direction = direction
        self.reverse = reverse_time
        data, time = self.model(self.direction, reverse_time = self.reverse)
        self.time = data.x
        self.data = data.y
        self.directory = directory

    def create_volume_envelope(self, t, vx):
        """
        Create amplitude envelope for velocity sign. Negative corresponds to left and positive
        corresponds to right.
        
        Parameters
        ----------
        t : ndarray
        vx : ndarray
            Velocity along one direction. This should be a one-dimesnional array.

        Returns
        -------
        ampLeft : ndarray
        ampRight : ndarray
        """
        smooth_vx = fftconvolve(vx,np.ones(12)/12.0,mode="same")
        smooth_vx = fftconvolve(smooth_vx[::-1],np.ones(12)/12.0,mode="same")[::-1]

        if self.direction.lower() == 'left': 
            smooth_vx = smooth_vx[::-1]

        smoothed_vel_x = itp.interp1d(np.arange(len(smooth_vx))/60.0,smooth_vx)

        xVelocity = smoothed_vel_x(t)

        ampLeft = np.exp(-np.sign(xVelocity)*np.abs(xVelocity*10))
        ampRight = np.exp(np.sign(xVelocity)*np.abs(xVelocity*10))
        
        max_xvel = max(max(ampLeft),max(ampRight))        
        ampLeft = ampLeft/max_xvel
        ampRight = ampRight/max_xvel

        ampLeft[ampLeft<0.1]=0.1
        ampRight[ampRight<0.1]=0.1        

        return ampLeft, ampRight


    def save_sound(self, filename, key,
                   use_stereo=False,
                   duration=None,
                   min_freq=150.,
                   max_freq=340.):
        """Convert motion into WAV audio file.

        Parameters
        ----------
        filename : str
        key : str
        use_stereo : bool, False
        duration : float, None
        min_freq : float,150.
        max_freq : float,340.
          
        Returns
        -------
        None
        """
        # Read in and filter motion data.
        data_amp = np.linalg.norm(self.data[:,1:],axis=1)
        smooth_data = fftconvolve(data_amp,np.ones(12)/12.0,mode='same')
        smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0,mode='same')[::-1]

        # Interpolation
        smoothed_vel = itp.interp1d(self.time,smooth_data)
        if duration is None:
            duration=self.time[-1]

        if not os.path.exists(self.directory) : os.mkdir(self.directory)

        freq_transform, factor = self.function_dict[key]
        print("freq_transform %s with factor %f"%(key,factor))

        # Revision for speed and correct scaling of frequency.
        t=np.arange(int(duration*WAV_SAMPLERATE))/WAV_SAMPLERATE/2
        rawVelocity=smoothed_vel(t)
        freq=freq_transform(rawVelocity*factor/10)
        # Rescale frequency to be between min max.
        freq=(freq-freq.min())/(freq.max()-freq.min())*(max_freq-min_freq)+min_freq
        phase=np.cumsum(freq)/WAV_SAMPLERATE
        amp=np.sin(phase*2.0*np.pi)
        
        # opening wav file that we will write to
        print("creating file with %f seconds"%duration)
        wavef = wave.open(os.path.join(self.directory,'_'.join([filename,key,self.direction])+'.wav'),'w')
        wavef.setnchannels(2)
        wavef.setsampwidth(2) 
        wavef.setframerate(WAV_SAMPLERATE)
        
        # Create sound file.
        if use_stereo:
            ampLeft, ampRight = self.create_volume_envelope(t,self.data[self.time<duration,1])
        else:
            ampLeft = ampRight = np.ones(len(t))
        assert len(ampLeft)==len(ampRight)==len(amp),(len(ampLeft),len(ampRight),len(amp))

        for fl, fr, a in zip(ampLeft, ampRight, amp):
            wavef.writeframesraw(struct.pack('<hh',a*fl*32767,a*fr*32767))

        wavef.writeframes('')
        wavef.close()    


if __name__ == '__main__':
    """
    This will run the code for turning velocity data into a sound file. 
    
    Call description:
    python soundplayer.py [TRANSFORM] [DR] [MODEL] [HAND] [REV] [STEREO] [DUR]

    Example call:
    python soundplayer_rom.py exp /Users/saeromchoi/Downloads 3_3 Right False True
    """
    mapType=sys.argv[1]
    dr=sys.argv[2]
    motionFileName=sys.argv[3]
    hand=sys.argv[4]
    reverseTime=True if sys.argv[5]=='True' else False
    useStereo=True if sys.argv[6]=='True' else False
    if len(sys.argv)>7:
        duration=float(sys.argv[7])
    else:
        duration=None
    assert mapType in ('exp','log','sig')
    assert os.path.isdir(dr)

    sp=SoundPlayer(dr,motionFileName,hand,reverseTime)
    sp.save_sound("motion_to_audio",mapType,True,duration)

