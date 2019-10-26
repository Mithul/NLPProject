import glob
import numpy as np
import python_speech_features as feats
import scipy.io.wavfile as wav

def extract_fbank(wavfile):
	fs, raw = wav.read(wavfile)
	fbank = feats.logfbank(raw,samplerate=fs,nfilt=40)

	return fbank

if __name__ == '__main__':
	features = []
	dtype = 'dev'
	path = '/Users/shobhanaganesh/Documents/NLP/project/en-de/data/'+dtype+'/wav/*'
	f = sorted(glob.glob(path))

	for i in range(len(f)):
		x = extract_fbank(f[i])
		features.append(x)

	features = np.asarray(features)

