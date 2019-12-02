import os, tqdm
import numpy as np
import python_speech_features as feats
import scipy.io.wavfile as wav

from collections import defaultdict

import torch

EOS_token = 1
PAD_token = 2
SPACE_token = 3
UNK_token = 4

def extract_fbank(wavfile):
	fs, raw = wav.read(wavfile)
	fbank = feats.logfbank(raw,samplerate=fs,nfilt=40)

	return fbank

class CharLang:
	def __init__(self, name):
		self.name = name
		self.word2index = defaultdict(lambda: UNK_token)
		self.word2index[" "] = SPACE_token
		self.word2count = {}
		self.index2word = {0: "SOS", EOS_token: "EOS", PAD_token: "PAD", SPACE_token: " ", UNK_token: "UNK"}
		self.n_words = 5  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)
		return sentence

	def addWord(self, word):
		for char in word:
			if char not in self.word2index:
				self.word2index[char] = self.n_words
				self.word2count[char] = 1
				self.index2word[self.n_words] = char
				self.n_words += 1
			else:
				self.word2count[char] += 1

	def indexesFromSentence(self, sentence):
		# print(sentence)
		return [0,3] + [self.word2index[char] for char in sentence]

	def tensorFromSentence(self, sentence, device, length=None):
		indexes = self.indexesFromSentence(sentence)
		indexes.append(SPACE_token)
		indexes.append(EOS_token)
		if length:
			indexes = indexes + [0]*(length - len(indexes))
			return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

		return torch.tensor(indexes, dtype=torch.long, device=device)

	def get_sentence(self, indeces):
		chars = []
		for index in indeces:
			chars.append(self.index2word[index.item()])
			if index == PAD_token or index == EOS_token:
				break

		return ''.join(chars)

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = defaultdict(lambda: UNK_token)
		self.word2count = {}
		self.index2word = {0: "SOS", EOS_token: "EOS", PAD_token: "PAD", UNK_token: "UNK"}
		self.n_words = 4  # Count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

		return sentence

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	def indexesFromSentence(self, sentence):
		return [0] + [self.word2index[word] for word in sentence.split(' ')]

	def tensorFromSentence(self, sentence, device, length=None):
		indexes = self.indexesFromSentence(sentence)
		indexes.append(EOS_token)
		if length:
			indexes = indexes + [0]*(length - len(indexes))
			return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

		return torch.tensor(indexes, device=device)

	def get_sentence(self, indeces):
		words = []
		for index in indeces:
			words.append(self.index2word[index.item()])

		return ' '.join(words)

class WavData(object):
	def __init__(self, name):
		self.name = name
		self.n_words = 40

	def addFeats(self, feats):
		return feats

	def tensorFromSentence(self, feats, device):
		return torch.tensor(feats, dtype=torch.float, device=device)

	def get_sentence(self, data):
		return "WAV_DATA"

class MUSTCData(object):
	def __init__(self, l1, l2, dataset_type = "train", character_level=False):
		self.l1 = l1
		self.l2 = l2
		self.dataset_type = dataset_type
		self.character_level = character_level
		self.input_lang = WavData(l1)
		if self.character_level:
			self.output_lang = CharLang(l2)
		else:
			self.output_lang = Lang(l2)

	def get_batch(self, data_directory="./data", batch_size = 1, max_sent_len=6, max_frames=600, buffer_factor=8, sort_len=False, min_sent_len=1):
		data_directory = os.path.join(data_directory, self.l1+"-"+self.l2)
		feats = os.path.join(data_directory, "features/" + self.dataset_type + "/feats/feat.tokenized.tsv")
		batches = []
		batch = []

		def chunks(a, n):
			"""Yield successive n-sized chunks from l."""
			n = max(1, n)
			return (a[i:i+n] for i in range(0, len(a), n))

		for i, line in enumerate(open(feats)):
			if len(batch) >= batch_size:
				# yield batch
				batches.append(batch)
				batch = []
				if len(batches) >= buffer_factor:
					if not sort_len:
						for batch in batches:
							yield batch
					else:
						all_speech_feats = []
						all_sentences = []
						for batch in batches:
							for speech_feat, sentence in batch:
								all_speech_feats.append(speech_feat)
								all_sentences.append(sentence)

						# all_speech_feats = sorted(all_speech_feats, key=lambda d: len(d))
						# all_sentences = sorted(all_sentences, key=lambda d: len(d))
						# print(list(map(lambda d: len(d), all_speech_feats)))
						# print(list(map(lambda d: len(d), all_sentences)))

						combined_data = list(zip(all_speech_feats, all_sentences))
						combined_data = list(sorted(combined_data, key=lambda d: [len(d[1]), len(d[0])]))
						batches = chunks(combined_data, batch_size)
						for batch in batches:
							yield batch
						# print(i, len(all_speech_feats))

					batches = []
				batch = []

				# break

			l1_sentence, l2_sentence, featfile, _, _, _, l1_s, l2_tokenized_s = line.split("\t")
			if len(l2_tokenized_s.strip().lower().split(" ")) < max_sent_len and len(l2_tokenized_s.strip().lower().split(" ")) > min_sent_len:
				if ":" in set(l2_tokenized_s.strip().lower().split(" ")):
					continue
				featfile = os.path.join(data_directory, featfile)
				speech_feats = np.load(featfile)
				if len(speech_feats) < max_frames:
					batch.append([speech_feats, l2_tokenized_s.strip().lower()])

		if len(batch) > 0:
			yield batch

	def prepareData(self, data_directory="./data", reverse=False):
		data_directory = os.path.join(data_directory, self.l1+"-"+self.l2)
		words_file = os.path.join(data_directory, self.l2 + ".words")
		with open(words_file) as f:
			for line in f:
				word = line.strip()
				self.output_lang.addWord(word)
		return self.input_lang, self.output_lang, None

	def tensorsFromPair(self, pair, device):
		input_tensor = self.input_lang.tensorFromSentence(pair[0], device)
		target_tensor = self.output_lang.tensorFromSentence(pair[1], device)
		return (input_tensor, target_tensor)


if __name__ == '__main__':
	mc_data = MUSTCData('en', 'de', character_level=False)
	input_lang, output_lang, _ = mc_data.prepareData(data_directory="./")
	b = mc_data.get_batch(data_directory="./", batch_size=2)
	for batch in tqdm.tqdm(b):
		# print()
		# batch = next(b)
		for speech_feats, sentence in batch:
			print(speech_feats)
			print(sentence)
			# print(len(speech_feats), len(sentence.split(" ")))
			# print(output_lang.tensorFromSentence(sentence, 'cpu'))
			# print(output_lang.get_sentence(output_lang.tensorFromSentence(sentence, 'cpu')))
			pass
		break

	mc_dev_data = MUSTCData('en', 'de', dataset_type="dev", character_level=False)
	b_dev = mc_data.get_batch(data_directory="./", batch_size=2)
	for batch in tqdm.tqdm(b):
		for speech_feats, sentence in batch:
			print(speech_feats)
			print(sentence)
			pass
		break

	# i, o, p = yn_data.prepareData()
	# print(yn_data.tensorsFromPair(p[0], 'cuda')[0].size())
