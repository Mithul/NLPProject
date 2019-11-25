import baselineAlt
import numpy as np
import random, tqdm, os, sys

from nltk.translate.bleu_score import sentence_bleu

import torch
import torch.nn.functional as F
import torch.nn as nn

from config import DEBUG, device
from data.data import MUSTCData as Dataset
from data.data import PAD_token, UNK_token
from torch import optim

from torch.utils.tensorboard import SummaryWriter

DEBUG = False

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage : python evaluate.py model_path [char|word] [max_sentence_length(default:6)]")
		exit()

	mc_data_train = Dataset('en', 'de', dataset_type="train", character_level=True)
	input_lang, output_lang, _ = mc_data_train.prepareData()
	mc_data = Dataset('en', 'de', dataset_type="tst-COMMON", character_level=True)
	if DEBUG: print("DIM", output_lang.n_words)
	seq = baselineAlt.Seq2Seq(output_lang.n_words).to(device)
	seq.init_weights()
	print(f'The model has {seq.count_parameters():,} trainable parameters')

	SAVE_PATH = sys.argv[1]

	if os.path.exists(SAVE_PATH):
		checkpoint = torch.load(SAVE_PATH,map_location=torch.device('cpu'))
		state_dict = checkpoint['model_state_dict']
		for key, val in seq.state_dict().items():
			if key not in state_dict:
				print("Missing model params for", key, "will reinitialize layer with", val)
				state_dict[key] = val
		for key, val in list(state_dict.items()):
			if key not in seq.state_dict():
				del state_dict[key]

		seq.load_state_dict(state_dict)

		seq.eval()

		max_sent_len = 6
		level = 'char'
		if len(sys.argv) > 2:
			level = sys.argv[2]
		if len(sys.argv) > 3:
			max_sent_len = int(sys.argv[3])

		max_frames = max_sent_len * 160
		print("Running test for",SAVE_PATH,"with max_sent_len as",max_sent_len,"and max_frames",max_frames,"at",level,"level")

		count = 0
		total_score = 0
		with torch.no_grad():
			b = mc_data.get_batch(batch_size=32,buffer_factor=1, max_sent_len=max_sent_len, max_frames=max_frames)
			for speech_feats, sentence_feats in tqdm.tqdm(baselineAlt.get_batch(b, output_lang)):
			#for speech_feats, sentence_feats in mc_data.get_batch(batch_size=1):
				# if DEBUG: print("START")
				f = speech_feats
				trg= sentence_feats
				# print("f",f.size())
				# print("trg",trg.size())

				# if DEBUG: print("F", f.size())
				seq.eval()
				outputs, loss,_ = seq(f,trg)
				for output,trgt in zip(outputs,trg):
					if level=='char':
						out_sen = list(output_lang.get_sentence(output))
						grnd_sen = [list(output_lang.get_sentence(trgt))]
					elif level=='word':
						out_sen = output_lang.get_sentence(output).split()
						grnd_sen = [output_lang.get_sentence(trgt).split()]
					else:
						print("Level must be 'word' or 'char'")
						exit()
					# print(' '.join(out_sen), ' '.join(grnd_sen[0]))
					if "applaus" not in set(' '.join(grnd_sen[0]).split(" ") + ''.join(grnd_sen[0]).split(" ")) :
						score = sentence_bleu(grnd_sen,out_sen)
						count = count + 1
						total_score += score
					# else:
						# print("REMOVED")
					score = sentence_bleu(grnd_sen,out_sen)
					count = count + 1
					total_score += score
					# break
				# print(total_score/count)

		bleu = total_score/count

		print("BLEU score: ",bleu)
