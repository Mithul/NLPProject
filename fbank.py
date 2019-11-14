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

from multiprocessing.pool import Pool as Pool

def extract_file(wav_file,dtype,pbar=None):
	fbank = extract_fbank(wav_file,dtype)
	feat_file = '/Users/shobhanaganesh/Documents/NLP/project/en-de/features/'+dtype+'/'+wav_file.split('/')[-1]
	np.save(feat_file, fbank)
	if pbar:
		pbar.update(1)

if __name__ == '__main__':
	features = []

	#pool = Pool(4)

	#works = []
	pbar = tqdm.tqdm()

	tfiles = 0
	#dtype = 'dev'
	for dtype in ['dev','train','tst-COMMON','tst-HE']:
		wavpath = path+dtype+'/wav/*'
		wav_files = sorted(glob.glob(wavpath))
		tfiles += len(wav_files)
		for wav_file in wav_files:
			#works.append(pool.apply_async(extract_file, (wav_file, dtype, pbar)))
			 fbank = extract_fbank(wav_file,dtype)
			 feat_file = '/Users/shobhanaganesh/Documents/NLP/project/en-de/features/'+dtype+'/'+wav_file.split('/')[-1].split('.wav')[0]
			 np.save(feat_file, np.asarray(fbank))

	#print("Sent")
	#pbar.total = tfiles
	#for work in works:
	#	work.get()

	#pool.join()
	#pool.close()

