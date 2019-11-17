import os
import numpy as np
import python_speech_features as feats
import scipy.io.wavfile as wav

import torch

EOS_token = 1
SPACE_token = 3
PAD_token = 2

def extract_fbank(wavfile):
	fs, raw = wav.read(wavfile)
	fbank = feats.logfbank(raw,samplerate=fs,nfilt=40)

	return fbank

class CharLang:
	def __init__(self, name):
		self.name = name
		self.word2index = {" ": SPACE_token}
		self.word2count = {}
		self.index2word = {0: "SOS", EOS_token: "EOS", PAD_token: "PAD", SPACE_token: " "}
		self.n_words = 4  # Count SOS and EOS

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
		return [0,3] + [self.word2index[char] for char in sentence]

	def tensorFromSentence(self, sentence, device, length=None):
		indexes = self.indexesFromSentence(sentence)
		indexes.append(SPACE_token)
		indexes.append(EOS_token)
		if length:
			indexes = indexes + [0]*(length - len(indexes))
			return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

		return torch.tensor(indexes, device=device)

	def get_sentence(self, indeces):
		chars = []
		for index in indeces:
			chars.append(self.index2word[index.item()])

		return ''.join(chars)

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", EOS_token: "EOS", PAD_token: "PAD"}
		self.n_words = 3  # Count SOS and EOS

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
	def __init__(self, l1, l2, character_level=False):
		self.l1 = l1
		self.l2 = l2
		self.character_level = character_level
		self.input_lang = WavData(l1)
		if self.character_level:
			self.output_lang = CharLang(l2)
		else:
			self.output_lang = Lang(l2)

	def get_batch(self, data_directory="./data", batch_size = 1):
		batch = []
		data_directory = os.path.join(data_directory, self.l1+"-"+self.l2)
		feats = os.path.join(data_directory, "features/train/feats/feat.tokenized.tsv")
		for line in open(feats):
			batch_index = 0
			while batch_index < batch_size:
				l1_sentence, l2_sentence, featfile, _, _, _, l1_s, l2_tokenized_s = line.split("\t")
				featfile = os.path.join(data_directory, featfile)
				batch.append([np.load(featfile), l2_tokenized_s.strip().lower()])
				batch_index += 1
			yield batch

	def prepareData(self, data_directory="./", reverse=False):
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
	b = mc_data.get_batch(data_directory="./", batch_size=1)
	batch = next(b)
	for speech_feats, sentence in batch:
		print(speech_feats)
		print(sentence)
		print(output_lang.tensorFromSentence(sentence, 'cpu'))
		print(output_lang.get_sentence(output_lang.tensorFromSentence(sentence, 'cpu')))
	# i, o, p = yn_data.prepareData()
	# print(yn_data.tensorsFromPair(p[0], 'cuda')[0].size())
