import convlstmAlt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

from torch import optim

class Encoder(nn.Module):
	def __init__(self,input_dim,input_channels,hidden_dim,output_dim,n_layers):
		super(Encoder,self).__init__()
		self.input_dim = input_dim
		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.n_layers = n_layers
		self.conv1 = nn.Conv2d(self.input_channels,32,3,2,1)
		self.conv2 = nn.Conv2d(32,32,3,2,1)
		self.clstm = convlstmAlt.ConvLSTM(input_channels=32, hidden_channels=[256], kernel_size=(1,3))
		self.LSTM = nn.LSTM(32*20,self.hidden_dim,self.n_layers,bidirectional=True)
		self.fc1 = nn.Linear(self.output_dim,self.output_dim)

	def forward(self,input):
		#input = input.unsqueeze(0) 
		print(input.size())
		input = input.float()
		batch_size = input.size(0)
		x = self.conv1(input)
		print("shape1",x.size())
		x = self.conv2(x)
		print("shape2",x.size())
		x = x.permute(2,0,1,3)
		x = x.contiguous().view(x.size(0),x.size(1),-1)
		print("shape3",x.size())
		h = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)
		c = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim)
		outputs,(h,c) =  self.LSTM(x,(h,c))
		#print('sum',torch.sum(outputs[-1]-torch.cat((h[-2, :, :],h[-1, :, :]))))
		print("shapeOut",outputs.size())
		print("shapeH",h.size())
		print("shapeC",c.size())
		newOutput = []
		for out in outputs:
			newOutput.append(F.relu(self.fc1(out)))
		out = torch.stack(newOutput)
		print("shapeOut2",out.size())
		return out,h,c
		'''
		out = F.relu((self.fc1(h[-2, :, :] + h[-1, :, :])))
		print("shapeOut2",out.size())'''
		#batch normalization]



class Attention(nn.Module):
	def __init__(self,e_hidden_dim,d_hidden_dim):
		super(Attention,self).__init__()
		self.ae = nn.Linear(e_hidden_dim,128)
		self.ad = nn.Linear(d_hidden_dim,128)

	def forward(self,enc_outputs,dec_output):
		#decoder op = (batchxd_hidden_dim)
		#encoder ops = (Lxbatchxe_hidden_dim)
		print("enc_outputs",enc_outputs.size())
		print("dec_outputs",dec_output.size())
		alphas = []
		ad_ok = self.ad(dec_output)
		ad_ok = ad_ok.unsqueeze(2)
		print("ad-ok",ad_ok.size())
		
		for i in range(enc_outputs.size(0)):
			ae_hl = self.ae(enc_outputs[i,:,:])
			ae_hl = ae_hl.unsqueeze(1)
			#print("ae-hl",ae_hl.size())
			alphas.append(torch.bmm(ae_hl,ad_ok))
		alphas = torch.stack(alphas)
		alphas = alphas.squeeze(-1)
		alphas = alphas.squeeze(-1)
		print("alphas: ",alphas.size())
		alphas = nn.Softmax(alphas)
		alphas = alphas.dim
		alphas = alphas.permute(1,0)
		alphas = alphas.unsqueeze(1)
		print("alphas: ",alphas.size())
		enc_outputs = enc_outputs.permute(1,0,2)
		print("enc_outputs",enc_outputs.size())
		context = torch.zeros(enc_outputs.size(2))
		context = torch.bmm(alphas,enc_outputs)
		context = context.squeeze(1)
		print ("context", context.size())
		return context

class Decoder(nn.Module):
	def __init__(self,output_dim,dec_hidden_dim,enc_hidden_dim,attention_dim,n_layers):
		super(Decoder,self).__init__()
		self.output_dim = output_dim
		self.dec_hidden_dim = dec_hidden_dim
		self.attention_dim = attention_dim
		self.n_layers = n_layers
		self.first_LSTM = nn.LSTM(self.output_dim+self.attention_dim,self.dec_hidden_dim,1,bidirectional=False)
		self.LSTM = nn.LSTM(self.dec_hidden_dim+self.attention_dim,self.dec_hidden_dim,self.n_layers-1,bidirectional=False)
		self.hidden = None
		self.cell_state = None
		self.attn = Attention(enc_hidden_dim,dec_hidden_dim)
		self.linear = nn.Linear(self.dec_hidden_dim+self.attention_dim,self.output_dim )
		self.enc_outputs = None


	def forward(self,input):
		print("Input size ",input.size())
		concat_input = torch.cat((input,self.context),dim=1)
		concat_input = concat_input.unsqueeze(0)
		print("Concatenated size: ",concat_input.size())
		o1 , (h1,c1) = self.first_LSTM(concat_input,(self.hidden[0,:,:].unsqueeze(0),self.cell_state[0,:,:].unsqueeze(0)))
		o1 = o1.squeeze(0)
		print("o1 size ",o1.size(),"hidden size: ",self.hidden.size())
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
		print("out",out.size())
		return out


	def initStates(self,hidden,cell_state,enc_outputs):
		self.hidden = hidden
		self.cell_state = cell_state
		self.enc_outputs = enc_outputs
		zeroes = torch.zeros(self.enc_outputs.size(1),self.dec_hidden_dim)
		self.context = self.attn.forward(self.enc_outputs,zeroes)




class Seq2Seq:
	def __init__(self,src):
		self.encoder = Encoder(src.size(3),1,256,512,4)
		self.decoder = Decoder(64,256,512,512,4)
		self.loss = nn.CrossEntropyLoss()
		self.enc_optim = optim.Adam(self.encoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01, amsgrad=False)
		self.dec_optim = optim.Adam(self.decoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01, amsgrad=False)


	def train(self,src,trg):
		#src #batch x chn x n_frames x filters
		#trg #batch x trg_len x embedding_dim

		self.enc_optim.zero_grad()
		self.dec_optim.zero_grad()

		enc_outputs,enc_hidden,enc_cell_state = self.encoder.forward(src)
		print("hidden ",enc_hidden.size())
		dec_init_hidden = torch.zeros(4,enc_hidden.size(1),enc_hidden.size(2))
		dec_init_cell_state = torch.zeros(4,enc_hidden.size(1),enc_hidden.size(2))
		j = 0
		for i in range(0,enc_hidden.size(0),2):
			dec_init_hidden[j] = enc_hidden[i,:,:] + enc_hidden[i+1,:,:]
			dec_init_cell_state[j] = enc_cell_state[i,:,:] + enc_cell_state[i+1,:,:]
			j = j+1

		self.decoder.initStates(dec_init_hidden,dec_init_cell_state,enc_outputs)
		
		teacher_forcing_ratio = 1.0
		outputs=torch.zeros(trg.size(0),trg.size(1),trg.size(2))
		input = trg[:,0,:]
		print("target size: ",trg.size())
		loss = 0
		for t in range(1, trg.size(1)):

			#insert input token embedding, previous hidden state and all encoder hidden states
			#receive output tensor (predictions) and new hidden state
			#output, hidden = self.decoder(input, attention_context)
			output = self.decoder.forward(input)

			#place predictions in a tensor holding predictions for each token
			outputs[:,t,:] = output
			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio

			#get the highest predicted token from our predictions
			top1 = output.argmax(1)
			print("top1",top1,"output",output.size())
			print("outputs :",outputs.size(),"Targets: ",trg.size())

			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			input = trg[:,t,:] if teacher_force else top1

			loss += self.loss(output[1:,:],trg[1:,:].long())

		
		loss.backward()
		self.enc_optim.step()
		self.dec_optim.step()

		#return outputs


f = np.load('/Users/shobhanaganesh/Documents/NLP/project/en-de/features/dev/ted_767.npy')
f = torch.tensor(f)
qwe = f.size(0)
asd = f.size(1)
f = torch.cat((f,f,f))
f = f.view(3,1,qwe,asd)
trg = "HeutesprecheichzuIhnenuberEnergieundKlima".lower()
trg= list(trg)
tr=[]
for t in trg:
	tmp = [0]*64
	tmp[ord(t)-ord('a')]= 1
	tr.append(tmp)
trg = [tr[:],tr[:],tr[:]]
trg = torch.FloatTensor(trg)
print("trg",trg.size())

print("F", f.size())
seq = Seq2Seq(f)
seq.train(f,trg)

