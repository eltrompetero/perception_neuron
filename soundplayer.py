# =============================================================================================== #
# Module for writing mocap files to audio. 
# Author : Saerom Choi
# =============================================================================================== #
import os, wave, struct, math, sys, numpy as np, matplotlib.pyplot as plt
from pipeline import extract_motionbuilder_model3_3,extract_motionbuilder_Eddie_Grid_Model_2
from scipy.signal import fftconvolve
from scipy import interpolate as itp
from scipy.special import expit
import numpy.fft as fft
import pyaudio



def load_sound(filename):
	wav = wave.open(filename,'r')
	sig = wav.readframes('-1')
	sif = np.fromstring(sig,'int16')
	return sif


def save_sound(filename, freqlist, exp_samplerate, wav_samplerate=44100.0):
	'''
	Save frequency list into a wav sound file
	'''
	duration = 1.0/exp_samplerate # 1/samplerate seconds per each sample
	wavef = wave.open(filename,'w')
	wavef.setnchannels(1)
	wavef.setsampwidth(2) 
	wavef.setframerate(wav_samplerate)
	for frequency in freqlist:
		for i in range(int(duration * wav_samplerate)):
			value = np.int16(32767.0*math.cos(frequency*math.pi*float(i)/float(wav_samplerate))) # converting into PCM format
			data = struct.pack('<h', value)
			wavef.writeframesraw( data )
	wavef.writeframes('')
	wavef.close()




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

	


class SoundPlayer():
	def __init__(self, directory, model_name, direction, reverse_time):
		self.models = {'3_3': extract_motionbuilder_model3_3,
                               'Eddie_Grid_Model_2': extract_motionbuilder_Eddie_Grid_Model_2 }
		self.model = self.models[model_name]
		self.function_dict = {'exp': (np.exp, 0.75), 'log': (np.log,0.5), 'sig': (expit,0.5), '': (lambda x : x, 32.0)}
		self.direction = direction
		self.reverse = reverse_time
		data, time = self.model(self.direction, reverse_time = self.reverse)
		self.time = data.x
		self.data = data.y
		self.directory = directory



	def save_sound_leftright(self, filename, key, duration=None):
		positions = self.data
		moveRight = [dp[0] > 0 for dp in positions for x in range(44100/60)]


		data_amp = [np.linalg.norm(x) for x in self.data]
		smooth_data = fftconvolve(data_amp,np.ones(12)/12.0)
		smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0)[::-1]

		if self.direction.lower() == 'left': 
			smooth_data = smooth_data[::-1]
			moveRight = leftright[::-1]


		# Interpolation
		data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)
		

		if not os.path.exists(self.directory) : os.mkdir(self.directory)
		

	
		function, factor = self.function_dict[key]
		print("function %s with factor %f"%(key,factor))


		wav_samplerate = 44100.0

		# opening wav file
		# wavef = wave.open(os.path.join(SOUND_DIR,'_'.join([filename,key,self.direction])+'.wav'),'w')
		# wavef.setnchannels(2)
		# wavef.setsampwidth(2) 
		# wavef.setframerate(wav_samplerate)

		max_x = 5#len(smooth_data)/60 if duration is None else duration
		print("creating file with %f seconds"%max_x)

		# Scaling the original Range 
		# ex ) np.exp(min_value*factor)*2.0*np.pi

		myMinVal = function(min(smooth_data)*factor)
		myMaxVal = function(max(smooth_data)*factor)
		myRange = myMaxVal-myMinVal


		MIN_FREQ = 150.0
		MAX_FREQ = 1000.0

		step = 1.0/wav_samplerate
		x_val = 0.0
		y_cum = 0

		ind = 0

		yvals = []
		ycums = []
		yadds = []

		stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,channels=2,rate=44100,output=True)

		amps = []
		while x_val<max_x:
			y_val  = data_fx(x_val)
			y_add  = function(y_val*factor) #rescaled and transformed
			y_add  = (y_add-myMinVal)/myRange*(MAX_FREQ-MIN_FREQ)+MIN_FREQ # scale from one range to another
			y_cum += y_add/wav_samplerate # f(t)*t
			amp    = np.sin(y_cum*2.0*np.pi)
			print amp
			ind   += 1
			amps.append(amp)
			x_val += step

		stream.write(np.array(amps).astype(np.float32))
		wavef.writeframes('')
		wavef.close()	


	def save_sound(self, filename, key, duration=None):
                """Convert motion into WAV audio file.
                Parameters
                ----------
                filename : str
                key : str
                duration : float,None

                Returns
                -------
                None
                """
		data_amp = [np.linalg.norm(x) for x in self.data]
		smooth_data = fftconvolve(data_amp,np.ones(12)/12.0)
		smooth_data = fftconvolve(smooth_data[::-1],np.ones(12)/12.0)[::-1]

		if self.direction.lower() == 'left': 
			smooth_data = smooth_data[::-1]


		# Interpolation
		data_fx = itp.interp1d(np.arange(len(smooth_data))/60.0,smooth_data)
		

		if not os.path.exists(self.directory) : os.mkdir(self.directory)
		

	
		function, factor = self.function_dict[key]
		print("function %s with factor %f"%(key,factor))


		wav_samplerate = 44100.0

		# opening wav file
		wavef = wave.open(os.path.join(self.directory,'_'.join([filename,key,self.direction])+'.wav'),'w')
		wavef.setnchannels(2)
		wavef.setsampwidth(2) 
		wavef.setframerate(wav_samplerate)

		max_x = len(smooth_data)/60 if duration is None else duration
		print("creating file with %f seconds"%max_x)

		# Scaling the original Range 
		# ex ) np.exp(min_value*factor)*2.0*np.pi

		myMinVal = function(min(smooth_data)*factor)
		myMaxVal = function(max(smooth_data)*factor)
		myRange = myMaxVal-myMinVal


		MIN_FREQ = 150.0
		MAX_FREQ = 1000.0

		step = 1.0/wav_samplerate
		x_val = 0.0
		y_cum = 0

		ind = 0

		yvals = []
		ycums = []
		yadds = []

		while x_val<max_x:
			y_val  = data_fx(x_val)
			yvals.append(y_val)
			y_add  = function(y_val*factor) #rescaled and transformed
			y_add  = (y_add-myMinVal)/myRange*(MAX_FREQ-MIN_FREQ)+MIN_FREQ # scale from one range to another
			yadds.append(y_add)
			y_cum += y_add/wav_samplerate # f(t)*t
			ycums.append(y_cum)
			amp    = np.sin(y_cum*2.0*np.pi)
			x_val += step
			wavef.writeframesraw(struct.pack('<h',amp*32767))

		wavef.writeframes('')
		wavef.close()	


if __name__ == '__main__':
    """Example call:
    python exp /Users/eddie/Downloads Eddie_Grid_Model_2 Right False
    """
    mapType=sys.argv[1]
    dr=sys.argv[2]
    motionFileName=sys.argv[3]
    hand=sys.argv[4]
    reverseTime=sys.argv[5]
    assert mapType in ('exp','log','sig')
    assert os.path.isdir(dr)

    sp = SoundPlayer(dr,motionFileName,hand,reverseTime)
    sp.save_sound_leftright("motion_to_audio",mapType)

