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
#
# def extract_fbank(wavfile):
# 	fs, raw = wav.read(wavfile)
# 	fbank = feats.logfbank(raw,samplerate=fs,nfilt=40)
# 	delta = feats.delta(fbank,15)
# 	delta_delta = feats.delta(delta,15)
#
# 	features = torch.stack((fbank,delta,delta_delta))
#
# 	return features

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
	def __init__(self, name, delta=False, delta_delta=False):
		self.name = name
		self.n_words = 40
		self.delta = delta
		self.delta_delta = delta_delta
		if delta:
			self.n_words = 80
			if delta_delta:
				self.n_words = 120

	def addFeats(self, feats):
		return feats

	def get_feats(self, feat):
		if self.delta:
			delta = feats.delta(feat,15)
			if self.delta_delta:
				delta_delta = feats.delta(delta,15)
				feat = [feat, delta, delta_delta]
			else:
				feat = [feat, delta]

		return np.asarray(feat)

	def tensorFromSentence(self, feat, device):
		return torch.tensor(feat, dtype=torch.float, device=device)

	def get_sentence(self, data):
		return "WAV_DATA"

class MUSTCData(object):
	def __init__(self, l1, l2, dataset_type = "train", character_level=False, delta=False, delta_delta=False, data_directory="./data", device='cpu'):
		self.l1 = l1
		self.l2 = l2
		self.dataset_type = dataset_type
		self.character_level = character_level
		self.input_lang = WavData(l1, delta, delta_delta)
		if self.character_level:
			self.output_lang = CharLang(l2)
		else:
			self.output_lang = Lang(l2)

		self.unproc_data = []
		self.data_directory = data_directory
		self.device = device

	def _get_unproc_data_lines(self, data_directory="./data", min_sent_len=None, max_sent_len=None):
		data_directory = os.path.join(data_directory, self.l1+"-"+self.l2)
		feats = open(os.path.join(data_directory, "features/" + self.dataset_type + "/feats/feat.tokenized.tsv")).readlines()
		if min_sent_len is not None:
			feats = filter(lambda d: len(d.split("\t")[7].split(" ")) >= min_sent_len, feats)
		if max_sent_len is not None:
			feats = filter(lambda d: len(d.split("\t")[7].split(" ")) <= max_sent_len, feats)
		return list(feats)

	def __len__(self):
		return len(self.unproc_data)

	def __getitem__(self, index):
		data_directory = self.data_directory
		data_directory = os.path.join(data_directory, self.l1+"-"+self.l2)
		max_sent_len=6
		max_frames=600
		# buffer_factor=8
		# sort_len=False
		min_sent_len=1

		line = self.unproc_data[index]
		l1_sentence, l2_sentence, featfile, _, _, _, l1_s, l2_tokenized_s = line.split("\t")
		# if len(l2_tokenized_s.strip().lower().split(" ")) < max_sent_len and len(l2_tokenized_s.strip().lower().split(" ")) > min_sent_len:
		# 	if ":" in set(l2_tokenized_s.strip().lower().split(" ")):
		# 		continue
		featfile = os.path.join(data_directory, featfile)
		speech_feats = self.input_lang.get_feats(np.load(featfile))
		# 	if len(speech_feats) < max_frames:
		return [speech_feats, l2_tokenized_s.strip().lower()]

	def prepareData(self, min_sent_len=None, max_sent_len=None, reverse=False):
		data_directory = self.data_directory
		self.unproc_data = self._get_unproc_data_lines(data_directory, min_sent_len=min_sent_len, max_sent_len=max_sent_len)
		data_directory = os.path.join(data_directory, self.l1+"-"+self.l2)
		words_file = os.path.join(data_directory, self.l2 + ".words")
		with open(words_file) as f:
			for line in f:
				word = line.strip()
				self.output_lang.addWord(word)

		return self.input_lang, self.output_lang, None

	def tensorsFromPair(self, pair):
		input_tensor = self.input_lang.tensorFromSentence(pair[0], self.device)
		target_tensor = self.output_lang.tensorFromSentence(pair[1], self.device)
		return (input_tensor, target_tensor)


	def collater(self, batch):
		# print(batch)
		speech_batch = []
		sentence_batch = []
		for speech_feats, sentence in batch:
			speech_batch.append(torch.tensor(speech_feats, dtype=torch.float, device=self.device))
			x = self.output_lang.tensorFromSentence(sentence, self.device)
			sentence_batch.append(x)

		speech_batch = torch.nn.utils.rnn.pad_sequence(speech_batch, batch_first=True, padding_value=PAD_token)
		sentence_batch = torch.nn.utils.rnn.pad_sequence(sentence_batch, batch_first=True, padding_value=PAD_token)
		return speech_batch, sentence_batch

if __name__ == '__main__':
	mc_data = MUSTCData('en', 'de', character_level=False, data_directory="./")
	input_lang, output_lang, _ = mc_data.prepareData(min_sent_len=3, max_sent_len=10)
	# print(mc_data[1])
	dataloader = torch.utils.data.DataLoader(mc_data, batch_size=4, shuffle=True, num_workers=1, collate_fn=mc_data.collater)
	for sp, sen in tqdm.tqdm(dataloader):
		# pass
		print(sp.size(), sen.size())
		break
