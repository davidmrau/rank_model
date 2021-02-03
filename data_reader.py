import csv
import re
import torch
import numpy as np
import time
from transformers import BertTokenizerFast
from utils import BasicTokenizer, TensorDict

class DataReader(torch.utils.data.IterableDataset):

	def __init__(self, encoding, data_file, num_docs, multi_pass, id2q, id2d, MB_SIZE, token2id=None, max_length_query=None, max_length_doc=None, encoded=False):

		self.num_docs = num_docs
		self.doc_col = 2 if self.num_docs <= 1 else 1
		self.MB_SIZE = MB_SIZE
		self.multi_pass = multi_pass
		self.id2d = id2d
		self.encoded = encoded
		self.reader = open(data_file, mode='r', encoding="utf-8")
		self.reader.seek(0)
		self.id2q = id2q
		self.encoding = encoding
		self.max_length_query = max_length_query
		self.max_length_doc = max_length_doc
		if encoding == 'bert':
			self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		elif encoding == 'glove':
			self.tokenizer = BasicTokenizer(token2id)
		else:
			raise ValueError(f'{encoding}')

	def __iter__(self):
		#t_start= time.time()

		while True:
			features = {}
			features['labels'] = torch.ones(self.MB_SIZE, dtype=torch.float)
			features['meta'] = []
			batch_queries, batch_docs, batch_q_lengths, batch_d_lengths = list(), list(), list(), list()
			for i in range(self.MB_SIZE):
				row = self.reader.readline()
				if row == '':
					if self.multi_pass:
						self.reader.seek(0)
						row = self.reader.readline()
					else:
						# file end while testing: stop iter by returning None and set seek to file start
						print('Training file exhausted, read again...')
						self.reader.seek(0)
						return
				cols = row.split()
				q = self.id2q[cols[0]]
				ds = [self.id2d[cols[self.doc_col + i].strip()] for i in range(self.num_docs)]
				if any(x is None for x in ds) or q is None:
					continue
				batch_d_lengths.append([min(len(d), self.max_length_doc) if self.max_length_doc else len(d) for d in ds])
				batch_q_lengths.append(min(len(q), self.max_length_query) if self.max_length_query else len(q))
					
				batch_queries.append(q)
				batch_docs.append(ds)

				if self.num_docs == 1:
					features['meta'].append([cols[0], cols[2], float(cols[4])])
			#t_tok_doc = time.time()

			batch_docs = np.array(batch_docs)
			batch_d_lengths = np.array(batch_d_lengths)
			batch_q_lengths = np.array(batch_q_lengths)

			# TODO: change that tokenizer accepts tokenized text 
			if not self.encoded:
				if self.encoding == 'bert':
					features['encoded_input'] = [TensorDict(self.tokenizer(batch_queries, list(map(lambda l : l[i], batch_docs)) , padding=True, truncation='only_second', return_tensors="pt")) for i in range(self.num_docs)]
				elif self.encoding == 'glove':
					features['encoded_input'] = list()
					q_encoded, lengths_q = self.tokenizer(batch_queries, padding=True,  return_tensor=True, max_length=self.max_length_query, if_empty='<pad>')
					for i in range(self.num_docs):
						d_encoded, lengths_d = self.tokenizer(batch_docs[:, i], padding=True, return_tensor=True, max_length=self.max_length_doc, if_empty='<pad>')
						encoded_dict = TensorDict({'encoded_query': q_encoded, 'encoded_doc': d_encoded , 'lengths_q':  torch.LongTensor(lengths_q), 'lengths_d': torch.LongTensor(lengths_d),  'max_length_d': np.max(lengths_d), 'max_length_q': np.max(lengths_q) })	
						features['encoded_input'].append(encoded_dict)

			else:
				if self.encoding == 'bert':
					#token_type_ids segment 00000111
					#attention_mask 11111111000
					features['encoded_input'] = [self.tokenizer(batch_queries, list(map(lambda l : l[i], batch_docs)) , padding=True, truncation='only_second', return_tensors="pt") for i in range(self.num_docs)]

				elif self.encoding == 'glove':
					features['encoded_input'] = list()
					q_encoded  = self.tokenizer.wrap_batch(batch_queries, batch_q_lengths, padding=True,  return_tensor=True, max_length=self.max_length_query, if_empty='<pad>')
					for i in range(self.num_docs):
						d_encoded = self.tokenizer.wrap_batch(batch_docs[:, i], batch_d_lengths[:,i ],  return_tensor=True, padding=True, max_length=self.max_length_doc, if_empty='<pad>')
						encoded_dict = TensorDict({'encoded_query': q_encoded, 'encoded_doc': d_encoded , 'lengths_q':  torch.LongTensor(batch_q_lengths), 'lengths_d': torch.LongTensor(batch_d_lengths[:, i]), 'max_length_d' : np.max(batch_d_lengths[:, i]), 'max_length_q' : np.max(batch_q_lengths)})	
						features['encoded_input'].append(encoded_dict)


			#print(f'tok_docs time: {time.time() - t_tok_doc}.')	
			#print(f'get_batch time: {time.time() - t_start}.')
					
			yield features
	def collate_fn(self, batch):
		return batch
