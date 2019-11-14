import os
import numpy as np
import python_speech_features as feats
import scipy.io.wavfile as wav

import torch

EOS_token = 1

def extract_fbank(wavfile):
	fs, raw = wav.read(wavfile)
	fbank = feats.logfbank(raw,samplerate=fs,nfilt=40)

	return fbank

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
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
        print(indeces)
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

class YesNoData(object):
    def __init__(self, l1, l2,):
        self.input_lang = WavData('yes_no_wavs')
        self.output_lang = Lang('yes_no_text')

    def prepareData(self, data_directory="data/waves_yesno", reverse=False):
        pairs = []
        for filename in os.listdir(data_directory):
            words = []
            if filename.endswith(".wav"):
                feats = extract_fbank(os.path.join(data_directory,filename))
                feats = self.input_lang.addFeats(feats)
                words = ["yes" if p == "1" else "no" for p in filename.split(".")[0].split("_")]
                words = self.output_lang.addSentence(' '.join(words))
                pairs.append([feats, words])

        self.pairs = pairs
        # print(pairs)
        return self.input_lang, self.output_lang, self.pairs

    def tensorsFromPair(self, pair, device):
        input_tensor = self.input_lang.tensorFromSentence(pair[0], device)
        target_tensor = self.output_lang.tensorFromSentence(pair[1], device)
        return (input_tensor, target_tensor)


if __name__ == '__main__':
    yn_data = YesNoData()
    i, o, p = yn_data.prepareData("waves_yesno")
    print(yn_data.tensorsFromPair(p[0], 'cuda')[0].size())
