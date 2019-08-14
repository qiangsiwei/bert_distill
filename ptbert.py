# -*- coding: utf-8 -*-

import os, csv, random, torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id

class Processor(object):
	def get_train_examples(self, data_dir):
		return self._create_examples(os.path.join(data_dir,'train.txt'),'train')
	def get_test_examples(self, data_dir):
		return self._create_examples(os.path.join(data_dir,'test.txt'),'test')
	def get_dev_examples(self, data_dir):
		return self._create_examples(os.path.join(data_dir,'dev.txt'),'dev')
	def get_labels(self):
		return ['0','1']
	def _create_examples(self, data_path, set_type):
		examples = []
		with open(data_path) as f:
			for i,line in enumerate(f):
				label,text = line.strip().split('\t',1)
				guid = "{0}-{1}-{2}".format(set_type,label,i)
				examples.append(InputExample(guid=guid,text=text,label=label))
		random.shuffle(examples)
		return examples

def convert_examples_to_features(examples, label_list, max_seq, tokenizer):
	label_map = {label:i for i,label in enumerate(label_list)}
	features = []
	for ex_index,example in enumerate(examples):
		tokens = tokenizer.tokenize(example.text)
		tokens = ["[CLS]"]+tokens[:max_seq-2]+["[SEP]"]
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1]*len(input_ids)
		padding = [0]*(max_seq-len(input_ids))
		label_id = label_map[example.label]
		features.append(InputFeatures(
			input_ids=input_ids+padding,
			input_mask=input_mask+padding,
			label_id=label_id))
	return features

class BertClassification(BertPreTrainedModel):
	def __init__(self, config, num_labels=2):
		super(BertClassification,self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)
	def forward(self, input_ids, input_mask, label_ids):
		_,pooled_output = self.bert(input_ids,None,input_mask,output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)
		if label_ids is not None:
			loss_fct = CrossEntropyLoss()
			return loss_fct(logits.view(-1,self.num_labels),label_ids.view(-1))
		return logits

class BertTextCNN(BertPreTrainedModel):
	def __init__(self, config, hidden_size=128, num_labels=2):
		super(BertTextCNN,self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.conv1 = nn.Conv2d(1,hidden_size,(3,config.hidden_size))
		self.conv2 = nn.Conv2d(1,hidden_size,(4,config.hidden_size))
		self.conv3 = nn.Conv2d(1,hidden_size,(5,config.hidden_size))
		self.classifier = nn.Linear(hidden_size*3,num_labels)
		self.apply(self.init_bert_weights)
	def forward(self, input_ids, input_mask, label_ids):
		sequence_output,_ = self.bert(input_ids,None,input_mask,output_all_encoded_layers=False)
		out = self.dropout(sequence_output).unsqueeze(1)
		c1 = torch.relu(self.conv1(out).squeeze(3))
		p1 = F.max_pool1d(c1,c1.size(2)).squeeze(2)
		c2 = torch.relu(self.conv2(out).squeeze(3))
		p2 = F.max_pool1d(c2,c2.size(2)).squeeze(2)
		c3 = torch.relu(self.conv3(out).squeeze(3))
		p3 = F.max_pool1d(c3,c3.size(2)).squeeze(2)
		pool = self.dropout(torch.cat((p1,p2,p3),1))
		logits = self.classifier(pool)
		if label_ids is not None:
			loss_fct = CrossEntropyLoss()
			return loss_fct(logits.view(-1,self.num_labels),label_ids.view(-1))
		return logits

def compute_metrics(preds, labels):
	return {'ac':(preds==labels).mean(),'f1':f1_score(y_true=labels,y_pred=preds)}

def main(bert_model='bert-base-chinese', cache_dir='/tmp/data/',\
	max_seq=128, batch_size=32, num_epochs=10, lr=2e-5):
	processor = Processor()
	train_examples = processor.get_train_examples('data/hotel')
	label_list = processor.get_labels()
	tokenizer = BertTokenizer.from_pretrained(bert_model,do_lower_case=True)
	model = BertClassification.from_pretrained(bert_model,\
		cache_dir=cache_dir,num_labels=len(label_list))
	# model = BertTextCNN.from_pretrained(bert_model,\
	# 	cache_dir=cache_dir,num_labels=len(label_list))
	model.to(device)
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params':[p for n,p in param_optimizer if not\
			any(nd in n for nd in no_decay)],'weight_decay':0.01},
		{'params':[p for n,p in param_optimizer if\
			any(nd in n for nd in no_decay)],'weight_decay':0.00}]
	print('train...')
	num_train_steps = int(len(train_examples)/batch_size*num_epochs)
	optimizer = BertAdam(optimizer_grouped_parameters,lr=lr,warmup=0.1,t_total=num_train_steps)
	train_features = convert_examples_to_features(train_examples,label_list,max_seq,tokenizer)
	all_input_ids = torch.tensor([f.input_ids for f in train_features],dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in train_features],dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in train_features],dtype=torch.long)
	train_data = TensorDataset(all_input_ids,all_input_mask,all_label_ids)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)
	model.train()
	for _ in trange(num_epochs,desc='Epoch'):
		tr_loss = 0
		for step,batch in enumerate(tqdm(train_dataloader,desc='Iteration')):
			input_ids,input_mask,label_ids = tuple(t.to(device) for t in batch)
			loss = model(input_ids,input_mask,label_ids)
			loss.backward(); optimizer.step(); optimizer.zero_grad()
			tr_loss += loss.item()
		print('tr_loss',tr_loss)
	print('eval...')
	eval_examples = processor.get_dev_examples('data/hotel')
	eval_features = convert_examples_to_features(eval_examples,label_list,max_seq,tokenizer)
	eval_input_ids = torch.tensor([f.input_ids for f in eval_features],dtype=torch.long)
	eval_input_mask = torch.tensor([f.input_mask for f in eval_features],dtype=torch.long)
	eval_label_ids = torch.tensor([f.label_id for f in eval_features],dtype=torch.long)
	eval_data = TensorDataset(eval_input_ids,eval_input_mask,eval_label_ids)
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data,sampler=eval_sampler,batch_size=batch_size)
	model.eval()
	preds = []
	for batch in tqdm(eval_dataloader,desc='Evaluating'):
		input_ids,input_mask,label_ids = tuple(t.to(device) for t in batch)
		with torch.no_grad():
			logits = model(input_ids,input_mask,None)
			preds.append(logits.detach().cpu().numpy())
	preds = np.argmax(np.vstack(preds),axis=1)
	print(compute_metrics(preds,eval_label_ids.numpy()))
	torch.save(model,'data/cache/model')

if __name__ == '__main__':
	main()
