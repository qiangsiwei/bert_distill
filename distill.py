# -*- coding: utf-8 -*-

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from ptbert import *
from small import *

FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class Teacher(object):
	def __init__(self, bert_model='bert-base-chinese', max_seq=128):
		self.max_seq = max_seq
		self.tokenizer = BertTokenizer.from_pretrained(
			bert_model,do_lower_case=True)
		self.model = torch.load('data/cache/model')
		self.model.eval()
	def predict(self, text):
		tokens = self.tokenizer.tokenize(text)[:self.max_seq]
		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1]*len(input_ids)
		padding = [0]*(self.max_seq-len(input_ids))
		input_ids = torch.tensor([input_ids+padding],dtype=torch.long).to(device)
		input_mask = torch.tensor([input_mask+padding],dtype=torch.long).to(device)
		logits = self.model(input_ids,input_mask,None)
		return F.softmax(logits,dim=1).detach().cpu().numpy()

if __name__ == '__main__':
	teacher = Teacher()
	# print(teacher.predict('还不错！这个价位算是物有所值了！'))

	import pickle
	from tqdm import tqdm
	x_len = 50; b_size = 64; lr = 0.001; epochs=30
	name = 'hotel' # clothing, fruit, hotel, pda, shampoo
	(x_tr,y_tr,t_tr),(x_de,y_de,t_de),(x_te,y_te,t_te),v_size,embs = load_data(name)
	l_tr = list(map(lambda x:min(len(x),x_len),x_tr))
	l_de = list(map(lambda x:min(len(x),x_len),x_de))
	l_te = list(map(lambda x:min(len(x),x_len),x_te))
	x_tr = sequence.pad_sequences(x_tr,maxlen=x_len)
	x_de = sequence.pad_sequences(x_de,maxlen=x_len)
	x_te = sequence.pad_sequences(x_te,maxlen=x_len)
	# t_tr = np.vstack([teacher.predict(text) for text in tqdm(t_tr)])
	# t_de = np.vstack([teacher.predict(text) for text in tqdm(t_de)])
	# with open('data/cache/t_tr','wb') as fout: pickle.dump(t_tr,fout)
	# with open('data/cache/t_de','wb') as fout: pickle.dump(t_de,fout)
	with open('data/cache/t_tr','rb') as fin: t_tr = pickle.load(fin)
	with open('data/cache/t_de','rb') as fin: t_de = pickle.load(fin)

	model = RNN(v_size,256,256,2)
	# model = CNN(v_size,256,128,2)
	if USE_CUDA: model = model.cuda()
	opt = optim.Adam(model.parameters(),lr=lr)
	loss1 = nn.NLLLoss()
	loss2 = nn.MSELoss()
	for epoch in range(epochs):
		losses = []; accu = []
		model.train()
		for i in range(0,len(x_tr),b_size):
			model.zero_grad()
			bx = Variable(LTensor(x_tr[i:i+b_size]))
			by = Variable(LTensor(y_tr[i:i+b_size]))
			bl = Variable(LTensor(l_tr[i:i+b_size]))
			bt = Variable(FTensor(t_tr[i:i+b_size]))
			py1,py2 = model(bx,bl)
			loss = loss1(py2,by)+loss2(py1,bt) # in paper, only mse is used
			loss.backward(); opt.step()
			losses.append(loss.item())
		for i in range(0,len(x_de),b_size):
			model.zero_grad()
			bx = Variable(LTensor(x_de[i:i+b_size]))
			bl = Variable(LTensor(l_de[i:i+b_size]))
			bt = Variable(FTensor(t_de[i:i+b_size]))
			py1,py2 = model(bx,bl)
			loss = loss2(py1,bt)
			loss.backward(); opt.step()
			losses.append(loss.item())
		model.eval()
		with torch.no_grad():
			for i in range(0,len(x_de),b_size):
				bx = Variable(LTensor(x_de[i:i+b_size]))
				by = Variable(LTensor(y_de[i:i+b_size]))
				bl = Variable(LTensor(l_de[i:i+b_size]))
				_,py = torch.max(model(bx,bl)[1],1)
				accu.append((py==by).float().mean().item())
		print(np.mean(losses),np.mean(accu))
