import glob
import numpy as np
import os
import python_speech_features as feats
import scipy.io.wavfile as wav
import sox
import tqdm
import yaml
import tempfile
from multiprocessing import Queue, Manager
from multiprocessing.pool import Pool

path = 'data/en-de/data/'

pool = None

def trim_wavfile(wavfile, start, end):
	transformer = sox.Transformer()
	tmpfile = tempfile._get_default_tempdir() + "/" + next(tempfile._get_candidate_names()) + ".wav"
	transformer.trim(start, end)
	transformer.build(wavfile, tmpfile)
	return tmpfile

def get_wavpath_from_segment(dataset_dir, segment):
	# dataset_type, _, yml_filename = yaml_file.split("/")[-3:]
	wavs_dir = os.path.join(dataset_dir, "wav")
	wavfile = os.path.join(wavs_dir, segment['wav'])
	return wavfile

def extract_fbank_from_wav(wavfile, start=0, duration=None):
	end = None
	if duration is not None:
		end = start + duration

	trimmed_wavfile = trim_wavfile(wavfile, start, end)
	fs, raw = wav.read(trimmed_wavfile)
	fbank = feats.logfbank(raw, samplerate=fs, nfilt=40)

	os.remove(trimmed_wavfile)
	return fbank

def dispatch_fbank_from_wav(wavfile, start=0, duration=None, queue=None, info=None):
	fbank = extract_fbank_from_wav(wavfile, start, duration)
	queue.put([fbank, info])
	# return fbank, info


def extract_fbanks_from_yaml(yaml_file):
	global pool

	features = []
	segment_path = yaml_file
	segment_data = yaml.load(open(segment_path), Loader=yaml.FullLoader)
	segments = segment_data #map(lambda d: json.loads(d), segment_data)

	jobs = []
	# m = Manager()
	# queue = m.Queue()
	for index, segment in enumerate(tqdm.tqdm(segments)):
		# if segment['wav'] == fname:
		dataset_dir = '/'.join(yaml_file.split("/")[:-2])
		wavfile = get_wavpath_from_segment(dataset_dir, segment)
		start = float(segment['offset'])
		duration = float(segment['duration'])

		# MultiProc
		# jobs.append(pool.apply_async(dispatch_fbank_from_wav, (wavfile, start, duration, queue, index)))

		# SingleProc
		fbank = extract_fbank_from_wav(wavfile, start, duration)
		yield fbank, wavfile.split('/')[-1], start, duration

	# for job in tqdm.tqdm(jobs):
		# job.get()

	# while not queue.empty():
		# features.append(queue.get())
		# queue.task_done()

	# queue.join()

	# features = sorted(features, key=lambda d: d[1])

	# return features

def get_sentences(text_dir, lang):
	dataset_type = text_dir.split('/')[-2]
	l1_sentences = []
	with open(os.path.join(text_dir, dataset_type + "." + lang)) as l1_file:
		for line in l1_file:
			yield line.strip()
		# l1_sentence = l1_file.readline()
	# return l1_sentences, l2_sentences

def save_sentence_and_features(feat_file, feat_dir, l1_sentence, l2_sentence, fbank):
	print(l1_sentences, l2_sentences, fbank)
	np.save(feat_file, np.asarray(fbank))


if __name__ == '__main__':
	features = []
	tfiles = 0
	lang1, lang2 = 'en', 'de'


	for dtype in tqdm.tqdm(['dev', 'train', 'tst-COMMON', 'tst-HE']):
		yamlpath = path+dtype+'/txt/*.yaml'
		yaml_files = sorted(glob.glob(yamlpath))
		tfiles += len(yaml_files)
		for yaml_file in yaml_files:
			text_dir = '/'.join(yaml_file.split("/")[:-1])
			l1_sentences = get_sentences(text_dir, lang1)
			l2_sentences = get_sentences(text_dir, lang2)
			fbanks = extract_fbanks_from_yaml(yaml_file)

			# print(len(list(l1_sentences)), len(list(l2_sentences)), len(list(fbanks)))

			feat_dir = '/'.join(path.split('/')[:-2])+'/features/'+dtype+'/feats'
			feat_file = os.path.join(feat_dir, 'feat.tsv')
			os.makedirs(feat_dir, exist_ok=True)
			with open(feat_file, 'w') as out_file:
				index = 0
				for l1_sentence, l2_sentence, fbank_data in zip(l1_sentences, l2_sentences, fbanks):
					# sentence, fbank = data
					fbank, wavfile, start, duration = fbank_data
					fbank_file = os.path.join(feat_dir, str(index) +".feat.npy")
					np.save(fbank_file, np.asarray(fbank))
					relative_fbank_path = '/'.join(fbank_file.split('/')[-4:])
					out_file.write(l1_sentence + "\t" + l2_sentence + "\t" + relative_fbank_path + "\t" + wavfile + "\t" + str(start) + "\t" + str(duration) + "\n")
					# save_sentence_and_features(out_file, feat_dir, l1_sentence, l2_sentence, fbank)
					index += 1
			# np.save(feat_file, np.asarray(fbank))
