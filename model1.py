import numpy as np
import random, tqdm, os

import torch
import torch.nn.functional as F
import torch.nn as nn

from config import DEBUG, device
from data.data import MUSTCData as Dataset
from data.data import PAD_token, UNK_token
from torch import optim

from torch.utils.tensorboard import SummaryWriter

from evaluate import get_bleu_score

DEBUG = False

class Encoder(nn.Module):
	def __init__(self,input_dim,input_channels,hidden_dim,output_dim,n_layers):
		super(Encoder,self).__init__()
		self.input_dim = input_dim
		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.n_layers = n_layers
		self.conv1 = nn.Conv2d(self.input_channels,32,3,2,1)
		self.bn = nn.BatchNorm2d(32)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32,32,3,2,1)
		# self.clstm = convlstmAlt.ConvLSTM(input_channels=32, hidden_channels=[256], kernel_size=(1,3))
		self.LSTM = nn.LSTM(32*10,self.hidden_dim,self.n_layers,bidirectional=True,dropout=0.3)
		self.LSTM_2 = nn.LSTM(512,self.hidden_dim,self.n_layers,bidirectional=True,dropout=0.3)
		self.fc1 = nn.Linear(self.output_dim,self.output_dim)
		self.fc2 = nn.Linear(32*10,self.output_dim)
		self.fc3 = nn.Linear(768,256)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self,input, h, c):
		input = input.unsqueeze(1)
		#input : [batch_size, channels(1), seq_len, feature_size]
		if DEBUG: print(input.size())
		# input = input.float()
		batch_size = input.size(0)
		x = self.conv1(input)
		x = self.bn(x)
		#x : [batch_size, channels(32), seq_len/2, feature_size]
		if DEBUG: print("shape1",x.size())
		x = self.conv2(x)
		x = self.bn2(x)
		#x : [batch_size, channels(32), seq_len/4, feature_size]
		if DEBUG: print("shape2",x.size())
		x = x.permute(2,0,1,3)
		#x : [seq_len/4, batch_size, channels(32), feature_size]
		x = x.contiguous().view(int(x.size(0)),x.size(1),-1)
		#x : [seq_len/4, batch_size, channels(32) x feature_size]
		if DEBUG: print("shape3",x.size())

		outputs,(h,c) =  self.LSTM(x,(h,c))
		#if DEBUG: print('sum',torch.sum(outputs[-1]-torch.cat((h[-2, :, :],h[-1, :, :]))))
		if DEBUG: print("shapeOut",outputs.size())
		if DEBUG: print("shapeH",h.size())
		if DEBUG: print("shapeC",c.size())

		h_op = outputs[-1] #ts x batch x 512
		if DEBUG:print("h_op size: ",h_op.size())
		h_op = h_op.unsqueeze(1)
		if DEBUG:print("h_op size: ",h_op.size())
		sa_ip = self.fc2(x.permute(1,0,2))
		sa_ip = sa_ip.permute(0,2,1)
		if DEBUG:print("sa_ip: ",sa_ip.size())
		energy = torch.tanh(torch.bmm(h_op,sa_ip)) #b x ts1 x ts2
		energy = energy.squeeze()
		if DEBUG:print("energy:",energy.size())
		attn_score = self.softmax(energy) # b x ts1
		if DEBUG:print("attn_score:",attn_score.size())
		attn_score = attn_score.unsqueeze(1)
		h_op = h_op.permute(0,2,1)
		if DEBUG:print(h_op.size())
		sa_context = torch.bmm(h_op,attn_score)
		if DEBUG:print("sa_context:",sa_context.size())
		h_op = h_op.permute(2,0,1)
		sa_context = sa_context.permute(2,0,1)
		lstm_ip = torch.cat((h_op,sa_context))
		if DEBUG:print("CONCATENATED:",lstm_ip.size())
		#lstm_input = self.fc3(torch.cat(h_op,sa_context).permute(1,0,2))


		outputs,(h,c) = self.LSTM_2(lstm_ip,(h,c))
		# newOutput = []
		# for out in outputs:
		# 	newOutput.append(F.relu(self.fc1(out)))
		# out = torch.stack(newOutput)
		out = F.relu(self.fc1(outputs))
		if DEBUG: print("shapeOut2",out.size())
		return out,h,c
		'''
		out = F.relu((self.fc1(h[-2, :, :] + h[-1, :, :])))
		if DEBUG: print("shapeOut2",out.size())'''
		#batch normalization]



class Attention(nn.Module):
	def __init__(self,e_hidden_dim,d_hidden_dim):
		super(Attention,self).__init__()
		self.ae = nn.Linear(e_hidden_dim,128)
		self.ad = nn.Linear(d_hidden_dim,128)
		self.softmax = nn.Softmax(dim=0)

	def forward(self,enc_outputs,dec_output):
		#decoder op = (batchxd_hidden_dim)
		#encoder ops = (Lxbatchxe_hidden_dim)
		if DEBUG: print("enc_outputs",enc_outputs.size())
		if DEBUG: print("dec_outputs",dec_output.size())
		alphas = []
		ad_ok = self.ad(dec_output)
		ad_ok = ad_ok.unsqueeze(2)
		if DEBUG: print("ad-ok",ad_ok.size())

		# for i in range(enc_outputs.size(0)):
		# 	ae_hl = self.ae(enc_outputs[i,:,:])
		# 	ae_hl = ae_hl.unsqueeze(1)
		# 	print("ae-hl", ad_ok.size(), ae_hl.size())
		# 	alphas.append(torch.bmm(ae_hl,ad_ok))
		ae_hl = self.ae(enc_outputs).unsqueeze(2)
		# repeated_ad_ok = torch.cat([ad_ok.unsqueeze(0)]*ae_hl.size(0), dim=0)
		if DEBUG : print("ae-hl", ad_ok.size(), ae_hl.size())
		alphas = torch.matmul(ae_hl, ad_ok)
		# alphas = torch.stack(alphas)
		if DEBUG : print(alphas.size())
		alphas = alphas.squeeze()
		# print("alphas: ",alphas.size())
		alphas = self.softmax(alphas)
		# alphas = alphas.dim
		alphas = alphas.permute(1,0)
		alphas = alphas.unsqueeze(1)
		if DEBUG: print("alphas: ",alphas.size())
		enc_outputs = enc_outputs.permute(1,0,2)
		if DEBUG: print("enc_outputs",enc_outputs.size())
		context = torch.bmm(alphas,enc_outputs)
		context = context.squeeze(1)
		if DEBUG: print ("context", context.size())
		return context

class Decoder(nn.Module):
	def __init__(self,output_dim,dec_hidden_dim,enc_hidden_dim,attention_dim,n_layers,input_dim):
		super(Decoder,self).__init__()
		self.embed_dim = 64
		self.embedding = nn.Embedding(input_dim, self.embed_dim)
		self.output_dim = output_dim
		self.dec_hidden_dim = dec_hidden_dim
		self.attention_dim = attention_dim
		self.n_layers = n_layers
		self.first_LSTM = nn.LSTM(self.embed_dim+self.attention_dim,self.dec_hidden_dim,1,bidirectional=False,dropout=0.3)
		self.LSTM = nn.LSTM(self.dec_hidden_dim+self.attention_dim,self.dec_hidden_dim,self.n_layers-1,bidirectional=False,dropout=0.3)
		self.hidden = None
		self.cell_state = None
		self.attn = Attention(enc_hidden_dim,dec_hidden_dim)
		self.linear = nn.Linear(self.dec_hidden_dim+self.attention_dim,self.output_dim )
		self.enc_outputs = None


	def forward(self,input):
		input = self.embedding(input)#.view(input.size(0), -1)
		if DEBUG: print("Input size ", input, input.size())
		if DEBUG: print("Context size ", self.context, self.context.size())
		concat_input = torch.cat((input,self.context),dim=1)
		concat_input = concat_input.unsqueeze(0)
		if DEBUG: print("Concatenated size: ",concat_input.size())
		first_h, first_c = self.hidden[0,:,:].unsqueeze(0),self.cell_state[0,:,:].unsqueeze(0)
		if DEBUG: print("Hiddens size: ",first_h.size(), first_c.size())
		o1 , (h1,c1) = self.first_LSTM(concat_input,(first_h, first_c ))
		o1 = o1.squeeze(0)
		if DEBUG: print("o1 size ",o1.size(),"hidden size: ",self.hidden.size())
		self.context = self.attn.forward(self.enc_outputs,o1)
		concat_input = torch.cat((o1,self.context),dim=1)
		concat_input = concat_input.unsqueeze(0)
		o, (h,c) = self.LSTM(concat_input,(self.hidden[1:,:,:],self.cell_state[1:,:,:]))
		self.hidden = torch.cat((h1,h))
		self.cell_state = torch.cat((c1,c))
		o = o.squeeze(0)
		concat_input = torch.cat((o,self.context),dim=1)
		out = self.linear(concat_input)
		#out = nn.Softmax(out)
		if DEBUG: print("out",out.size())
		return out


	def initStates(self,hidden,cell_state,enc_outputs, device):
		self.hidden = hidden
		self.cell_state = cell_state
		self.enc_outputs = enc_outputs
		zeroes = torch.zeros(self.enc_outputs.size(1),self.dec_hidden_dim, device=device)
		self.context = self.attn.forward(self.enc_outputs,zeroes)




class Seq2Seq(nn.Module):
	def __init__(self, target_dim):
		super(Seq2Seq, self).__init__()
		self.n_layers = 4
		self.hidden_dim = 256
		self.target_dim = target_dim
		self.encoder = Encoder(40, 1, self.hidden_dim, self.hidden_dim*2, self.n_layers) # FBANK_Feats, input_channels, hidden_dim, output_dim, n_layers
		self.decoder = Decoder(target_dim, self.hidden_dim, self.hidden_dim*2, self.hidden_dim*2, self.n_layers, target_dim) # output_dim, dec_hidden_dim, enc_hidden_dim, attention_dim, n_layers
		self.loss = nn.CrossEntropyLoss(ignore_index = PAD_token)
		# self.enc_optim = optim.Adam(self.encoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01, amsgrad=False)
		# self.dec_optim = optim.Adam(self.decoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01, amsgrad=False)


	def forward(self,src,trg, teacher_forcing_ratio = 0.8):
		#src #batch x chn x n_frames x filters
		#trg #batch x trg_len x embedding_dim

		# self.enc_optim.zero_grad()
		# self.dec_optim.zero_grad()

		batch_size = src.size(0)

		h = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim, device=device)
		c = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim, device=device)

		enc_outputs,enc_hidden,enc_cell_state = self.encoder.forward(src, h, c)
		# print(enc_outputs.size())
		if DEBUG: print("hidden ",enc_hidden.size())
		dec_init_hidden = torch.zeros(4,enc_hidden.size(1),enc_hidden.size(2), device=device)
		dec_init_cell_state = torch.zeros(4,enc_hidden.size(1),enc_hidden.size(2), device=device)
		j = 0
		for i in range(0,enc_hidden.size(0),2):
			dec_init_hidden[j] = enc_hidden[i,:,:] + enc_hidden[i+1,:,:]
			dec_init_cell_state[j] = enc_cell_state[i,:,:] + enc_cell_state[i+1,:,:]
			j = j+1

		self.decoder.initStates(dec_init_hidden,dec_init_cell_state,enc_outputs, device)


		outputs=torch.zeros(trg.size(1),trg.size(0), dtype=torch.long, device=device)
		input = trg[:,0]
		if DEBUG: print("target size: ",trg.size())
		loss = 0
		forced = []
		for t in range(1, trg.size(1)):

			#insert input token embedding, previous hidden state and all encoder hidden states
			#receive output tensor (predictions) and new hidden state
			#output, hidden = self.decoder(input, attention_context)
			output = self.decoder.forward(input)

			#place predictions in a tensor holding predictions for each token
			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio
			forced.append(teacher_force)

			#get the highest predicted token from our predictions
			topv, top1 = output.topk(1)
			top1 = top1.view(-1)
			outputs[t] = top1
			if DEBUG: print("top1",top1,"output",output.size())
			# if DEBUG: print("outputs :",outputs.size(),"Targets: ",trg.size())

			loss += self.loss(output[:,:],trg[:,t].long())

			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			input = trg[:,t] if teacher_force else top1.detach()
			if DEBUG: print("op/trg", output[1:,:].size(), trg[1:,:].size())

		trg = trg.permute(1,0)
		# print("OP", outputs.size(), trg.size())
		# print("OP", outputs, trg)
		# print(outputs[1:,:].view(-1, 1),trg[1:,:].contiguous().view(-1))
		# print(outputs[1:,:].view(-1, 1).size(),trg[1:,:].contiguous().view(-1).size())
		# loss += self.loss(outputs[1:,:].view(-1, 1),trg[1:,:].contiguous().view(-1))
		if DEBUG: print("LOSS", loss)

		# loss = loss/float(batch_size)
		# print("LOSS", loss)

		return outputs.permute(1, 0), loss, forced
		# loss.backward()
		# self.enc_optim.step()
		# self.dec_optim.step()

		#return outputs

	def count_parameters(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def init_weights(self):
	    def _init_weights(model):
	        for name, param in model.named_parameters():
	            if 'weight' in name:
	                nn.init.normal_(param.data, mean=0, std=0.01)
	            else:
	                nn.init.constant_(param.data, 0)
	    self.apply(_init_weights)

def get_batch(iterator, lang):
	for batch in iterator:
		speech_batch = []
		sentence_batch = []
		for speech_feats, sentence in batch:
			speech_batch.append(torch.tensor(speech_feats, dtype=torch.float, device=device))
			x = lang.tensorFromSentence(sentence, device)
			sentence_batch.append(x)

		speech_batch = torch.nn.utils.rnn.pad_sequence(speech_batch, batch_first=True, padding_value=PAD_token)
		sentence_batch = torch.nn.utils.rnn.pad_sequence(sentence_batch, batch_first=True, padding_value=PAD_token)
		yield speech_batch, sentence_batch


if __name__ == '__main__':
	mc_data = Dataset('en', 'de', character_level=True)
	input_lang, output_lang, _ = mc_data.prepareData()

	mc_dev_data = Dataset('en', 'de', dataset_type="dev", character_level=True)

	if DEBUG: print("DIM", output_lang.n_words)
	seq = Seq2Seq(output_lang.n_words).to(device)
	seq.init_weights()
	seq_optim = optim.Adam(seq.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.00001, amsgrad=False)
	print(f'The model has {seq.count_parameters():,} trainable parameters')

	writer = SummaryWriter("Self_attention_word")

	SAVE_PATH = "Self_attention_word.model"

	iter = 0

	iters_per_epoch = 0

	loss_checkpoint = 20000
	start_iter = None
	start_epoch = None

	if os.path.exists(SAVE_PATH):
		checkpoint = torch.load(SAVE_PATH,map_location=torch.device(device))
		state_dict = checkpoint['model_state_dict']
		for key, val in seq.state_dict().items():
			if key not in state_dict:
				print("Missing model params for", key, "will reinitialize layer with", val)
				state_dict[key] = val
		for key, val in list(state_dict.items()):
			if key not in seq.state_dict():
				del state_dict[key]

		optim_state_dict = checkpoint['optimizer_state_dict']
		print(optim_state_dict.keys())
		for key, val in seq_optim.state_dict().items():
			if key not in state_dict:
				print("Missing model params for", key, "will reinitialize layer with", val)
				optim_state_dict[key] = val

		for key, val in list(optim_state_dict.items()):
			if key not in seq_optim.state_dict():
				del optim_state_dict[key]

		seq.load_state_dict(state_dict)
		# seq_optim.load_state_dict(optim_state_dict)
		start_epoch = checkpoint['epoch']
		# start_iter = checkpoint['iter']
		loss = checkpoint['loss']
		loss_checkpoint = checkpoint['loss']
		iters_per_epoch = checkpoint['iters_per_epoch']
		print("Loaded", start_epoch, start_iter, loss, iters_per_epoch)


	b_dev = mc_dev_data.get_batch(batch_size=32, buffer_factor=1)
	dev_speech_feats, dev_sentence_feats = next(get_batch(b_dev, output_lang))

	REPEAT_TIMES = 5

	for epoch in range(1000):
		if start_epoch is not None:
			if start_epoch > epoch:
				continue
			if start_epoch == epoch:
				start_epoch = None

		iter = 0
		b = mc_data.get_batch(batch_size=96, max_sent_len=10, min_sent_len=4, max_frames=1000)

		for speech_feats, sentence_feats in get_batch(b, output_lang):
			for repeat in range(REPEAT_TIMES):
				print("ITER", iter)
				shuffled_indeces = torch.randperm(speech_feats.size(0))

				speech_feats = speech_feats[shuffled_indeces]
				sentence_feats = sentence_feats[shuffled_indeces]
				if start_iter is not None:
					if start_iter > iter:
						iter += 1
						continue
					if start_iter == iter:
						start_iter = None

				seq_optim.zero_grad()
				# if DEBUG: print(speech_feats)
				# if DEBUG: print(sentence)
				# if DEBUG: print(output_lang.tensorFromSentence(sentence, device))
				# if DEBUG: print(output_lang.get_sentence(output_lang.tensorFromSentence(sentence, device)))

				if DEBUG: print("START")
				f = speech_feats#torch.tensor(speech_feats, device=device)
				# f = f.view(3,1,qwe,asd)
				# qwe = f.size(0)
				# asd = f.size(1)
				# f = torch.cat((f,f,f))
				# trg = "HeutesprecheichzuIhnenuberEnergieundKlima".lower()
				trg= sentence_feats
				# tr=[]
				# for t in trg:
				# 	tmp = [0]*64
				# 	tmp[ord(t)-ord('a')]= 1
				# 	tr.append(tmp)
				# trg = [tr[:],tr[:],tr[:]]
				# trg = torch.FloatTensor(trg)
				print("f",f.size())
				print("trg",trg.size())

				if DEBUG: print("F", f.size())

				seq.train()
				outputs, loss, forced = seq(f,trg, teacher_forcing_ratio = (REPEAT_TIMES - repeat)/REPEAT_TIMES)


				loss.backward()
				torch.nn.utils.clip_grad_norm_(seq.parameters(), 1)
				seq_optim.step()

				seq.eval()
				dev_outputs, dev_loss, dev_forced = seq(dev_speech_feats, dev_sentence_feats, teacher_forcing_ratio = 0)
				print("LOSS", loss.item(), dev_loss.item())

				writer.add_scalar('Loss/train', loss, iters_per_epoch*epoch + iter)
				writer.add_scalar('Loss/dev', dev_loss, iters_per_epoch*epoch + iter)
				for output,trgt in zip(outputs[:4],trg[:4]):
					writer.add_text('output', output_lang.get_sentence(output), iters_per_epoch*epoch + iter)
					writer.add_text('target', output_lang.get_sentence(trgt), iters_per_epoch*epoch + iter)
					print("O", output_lang.get_sentence(output))
					print("T", output_lang.get_sentence(trgt))
					# print(forced)
					# break

				if iter%10 == 0 or (loss_checkpoint > loss.item()):
					for n, pr in seq.named_parameters():
						 if pr.requires_grad:
							 tag = "weights/"+n
							 writer.add_histogram(tag+"/raw", pr, iters_per_epoch*epoch + iter)
							 writer.add_scalar(tag+"/max", torch.max(pr).item(), iters_per_epoch*epoch + iter)
							 writer.add_scalar(tag+"/min", torch.min(pr).item(), iters_per_epoch*epoch + iter)
							 writer.add_scalar(tag+"/mean", torch.mean(pr).item(), iters_per_epoch*epoch + iter)
							 writer.add_scalar(tag+"/stddev", torch.std(pr).item(), iters_per_epoch*epoch + iter)

				if iter%50 == 0:
					writer.add_scalar('BLEU/char', get_bleu_score(dev_outputs, dev_sentence_feats, output_lang, bleu_level='char')[0], iters_per_epoch*epoch + iter)
					writer.add_scalar('BLEU/word', get_bleu_score(dev_outputs, dev_sentence_feats, output_lang, bleu_level='word')[0], iters_per_epoch*epoch + iter)

				if iter%50 == 0 or (loss_checkpoint > dev_loss.item() and iter%10 == 0):
					loss_checkpoint = dev_loss.item()
					torch.save({
			            'epoch': epoch,
			            'iter': iter,
						'iters_per_epoch': iters_per_epoch,
			            'model_state_dict': seq.state_dict(),
			            'optimizer_state_dict': seq_optim.state_dict(),
			            'loss': loss,
			            }, SAVE_PATH)
					# checkpoint = torch.load(SAVE_PATH)
					# seq.load_state_dict(checkpoint['model_state_dict'])
					# seq_optim.load_state_dict(checkpoint['optimizer_state_dict'])
					# epoch = checkpoint['epoch']
					# start_iter = checkpoint['iter']
					# loss = checkpoint['loss']
					# iters_per_epoch = checkpoint['iters_per_epoch']
					# print("Loaded", epoch, start_iter, loss, iters_per_epoch)

				# else: #TODO : Run loss on val set to check for improvement/restoring to previous state
				# 	if os.path.exists(SAVE_PATH):
				# 		checkpoint = torch.load(SAVE_PATH)
				# 		seq.load_state_dict(checkpoint['model_state_dict'])
				# 		seq_optim.load_state_dict(checkpoint['optimizer_state_dict'])
				# 		epoch = checkpoint['epoch']
				# 		start_iter = checkpoint['iter']
				# 		loss = checkpoint['loss']
				# 		iters_per_epoch = checkpoint['iters_per_epoch']
				# 		print("Loaded", epoch, start_iter, loss, iters_per_epoch)
				iter += 1

		iters_per_epoch = iter
