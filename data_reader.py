import csv
import re
import torch
import numpy as np

from transformers import BertTokenizerFast
from utils import BasicTokenizer, TensorDict

class DataReader():
	def __init__(self, embedding, data_file, num_docs, multi_pass, id2q, id2d, MB_SIZE, token2id=None, max_length=None):

		self.num_docs = num_docs
		self.doc_col = 2 if self.num_docs <= 1 else 1
		self.MB_SIZE = MB_SIZE
		self.multi_pass = multi_pass
		self.id2d = id2d
		self.reader = open(data_file, mode='r', encoding="utf-8")
		self.reader.seek(0)
		self.id2q = id2q
		self.embedding = embedding
		self.max_length = max_length
		if embedding is 'bert':
			self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		elif embedding is 'glove':
			self.tokenizer = BasicTokenizer(token2id)
		else:
			raise ValueError()




	def get_minibatch(self):
		features = {}
		features['labels'] = np.ones((self.MB_SIZE), dtype=np.int64)
		features['meta'] = []
		batch_queries, batch_docs = list(), list()
		for i in range(self.MB_SIZE):
			row = self.reader.readline()
			if row == '':
				if self.multi_pass:
					self.reader.seek(0)
					row = self.reader.readline()
				else:
					break
			cols = row.split()
			q = self.id2q[cols[0]]
			ds = [self.id2d[cols[self.doc_col + i].strip()] for i in range(self.num_docs)]
			for d in ds:
				if d is None:
					continue
			batch_queries.append(q)
			batch_docs.append(ds)	
			if self.num_docs == 1:
				features['meta'].append([cols[0], cols[2], float(cols[4])])
		if self.embedding is 'bert': 
			features['encoded_input'] = [self.tokenizer(batch_queries, list(map(lambda l : l[i], batch_docs)) , padding=True, truncation='only_second', return_tensors="pt") for i in range(self.num_docs)]
		elif self.embedding is 'glove':
			features['encoded_input'] = list()
			q_encoded, lengths_q = self.tokenizer(batch_queries, padding=True, return_tensor=True, max_length=self.max_length, if_empty='<pad>')
			for i in range(self.num_docs):
				d_encoded, lengths_d = self.tokenizer(list(map(lambda l : l[i], batch_docs)), padding=True, return_tensor=True, max_length=self.max_length, if_empty='<pad>')
				encoded_dict = TensorDict({'encoded_query': q_encoded, 'encoded_doc': d_encoded , 'lengths_q':  torch.LongTensor(lengths_q), 'lengths_d': torch.LongTensor(lengths_d) })		
				features['encoded_input'].append(encoded_dict)
		return features

	def reset(self):
		self.reader.seek(0)
