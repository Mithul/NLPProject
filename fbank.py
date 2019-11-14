import glob
import numpy as np
import os
import python_speech_features as feats
import scipy.io.wavfile as wav
import sox
import tqdm

path = '/Users/shobhanaganesh/Documents/NLP/project/en-de/data/'

def extract_fbank(wavfile,dtype):
	features = []
	segpath = path+dtype+'/txt/'+dtype+'.yaml'
	print(wavfile)
	segfile = open(segpath).readlines()
	fname = wavfile.split('/')[-1]
	for i in range(len(segfile)):
		if segfile[i].split()[-1].split('}')[0] == fname:
			start = float(segfile[i].split('offset:')[1].split(',')[0])
			duration = float(segfile[i].split('duration:')[1].split(',')[0])
			transformer = sox.Transformer()
			transformer.trim(start,start+duration)
			transformer.build(wavfile, 'out.wav')
			fs, raw = wav.read('out.wav')
			fbank = feats.logfbank(raw,samplerate=fs,nfilt=80)
			features.append(fbank)
			os.remove('out.wav')

	return fbank


if __name__ == '__main__':
	features = []

	pbar = tqdm.tqdm()

	tfiles = 0
	
	for dtype in ['dev','train','tst-COMMON','tst-HE']:
		wavpath = path+dtype+'/wav/*'
		wav_files = sorted(glob.glob(wavpath))
		tfiles += len(wav_files)
		for wav_file in wav_files:
			fbank = extract_fbank(wav_file,dtype)
			feat_file = path+'/features/'+dtype+'/'+wav_file.split('/')[-1].split('.wav')[0]
			np.save(feat_file, np.asarray(fbank))

	

