import csv
import re
import torch
import numpy as np
import time
from transformers import BertTokenizerFast
from utils import BasicTokenizer, TensorDict
import random

class DataReader(torch.utils.data.IterableDataset):
		

	def __init__(self, model, data_file, num_docs, multi_pass, id2q, id2d, MB_SIZE, token2id=None, max_length_query=None, max_length_doc=None, encoded=False, sample_random_docs=False, qrel_columns={'doc': 0, 'query': 2, 'score': 4}, encoding=None):

		self.num_docs = num_docs
		self.doc_col = 2 if self.num_docs <= 1 else 1
		self.MB_SIZE = MB_SIZE
		self.multi_pass = multi_pass
		self.id2d = id2d
		self.doc_ids = list(id2d.file.keys())
		self.encoded = encoded
		self.encoding = encoding
		self.reader = open(data_file, mode='r', encoding="utf-8")
		self.reader.seek(0)
		self.id2q = id2q
		self.model = model
		self.max_length_query = max_length_query
		self.max_length_doc = max_length_doc
		self.sample_random_docs = sample_random_docs
		self.qrel_columns = qrel_columns
		if 'bert' in model.lower():
			if encoded:
				self.tokenizer = BasicTokenizer(None)
			else:
				self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

		elif model == 'rank':
			if self.encoding == 'glove':
				self.tokenizer = BasicTokenizer(token2id)
			elif self.encoding == 'bert':
				self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		else:
			raise ValueError(f'{model}')
	def sample_rand_doc(self):
		rand = random.randint(0, len(self.doc_ids)-1)
		doc_id = self.doc_ids[rand]
		return doc_id


	def __iter__(self):
		#t_start= time.time()

		while True:
			features = {}
			features['labels'] = torch.ones(self.MB_SIZE, dtype=torch.float)
			features['meta'] = []
			features['encoded_input'] = list()
			batch_queries, batch_docs, batch_q_lengths, batch_d_lengths = list(), list(), list(), list()
			while len(batch_queries) < self.MB_SIZE:
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
				# get doc_ids	
				ds_ids = [cols[self.doc_col + i].strip() for i in range(self.num_docs)]
				# if sample_random_docs
				if self.num_docs == 2 and self.sample_random_docs:
					ds_ids[1] = self.sample_rand_doc()
			
				# get doc content
				ds = [self.id2d[id_] for id_ in ds_ids]
				# if any of the docs is None skip triplet	
				if any(x is None for x in ds) or q is None:
					continue

				batch_queries.append(q)
				batch_docs.append(ds)

				if self.num_docs == 1:
					#features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']], float(cols[self.qrel_columns['score']])])
					features['meta'].append([cols[self.qrel_columns['doc']], cols[self.qrel_columns['query']]])
			#t_tok_doc = time.time()

			batch_d_lengths = np.array(batch_d_lengths)
			batch_q_lengths = np.array(batch_q_lengths)

			# TODO: change that tokenizer accepts tokenized text 
			if not self.encoded:
				if self.model =='bert' :
					features['encoded_input'] = [TensorDict(self.tokenizer(batch_queries, [bd[i] for bd in batch_docs], padding=True, truncation='only_second', 
					return_tensors="pt",  max_length=self.max_length_doc)) for i in range(self.num_docs)]

				elif self.model == 'rank' and self.encoding == 'glove':
					q_encoded, lengths_q = self.tokenizer(batch_queries, padding=True,  return_tensor=True, max_length=self.max_length_query, if_empty='<pad>')
					for i in range(self.num_docs):
						d_encoded, lengths_d = self.tokenizer([bd[i] for bd in batch_docs], padding=True, return_tensor=True, max_length=self.max_length_doc, if_empty='<pad>')
						encoded_dict = TensorDict({'encoded_query': q_encoded, 'encoded_doc': d_encoded , 'lengths_q':  lengths_q, 'lengths_d': lengths_d})	
						features['encoded_input'].append(encoded_dict)
				elif self.model == 'rank' and self.encoding == 'bert':
					q_token = self.tokenizer(batch_queries, max_length=self.max_length_doc, padding=True, return_tensors="pt", add_special_tokens=False)
					d_lengths = (q_token.input_ids != 0).sum(1) 
					for i in range(self.num_docs):
						d_token = self.tokenizer([bd[i] for bd in batch_docs], padding=True, return_tensors="pt", max_length=self.max_length_doc, add_special_tokens=False)
						q_lengths = (d_token.input_ids != 0).sum(1) 
						encoded_dict = TensorDict({'encoded_query': q_token.input_ids, 'encoded_doc': d_token.input_ids , 'lengths_q':  q_lengths, 'lengths_d': d_lengths})	
						features['encoded_input'].append(encoded_dict)
				elif self.model == 'sparseBERT' :
					features['encoded_queries'] = TensorDict(self.tokenizer(batch_queries, padding=True, 
					return_tensors="pt", max_length=self.max_length_doc))

					features['encoded_docs'] = [TensorDict(self.tokenizer([bd[i] for bd in batch_docs], padding=True, 
					return_tensors="pt", max_length=self.max_length_doc)) for i in range(self.num_docs)]
				else:
					raise NotImplementedError()
			else:
				if self.model == 'bert':
					for i in range(self.num_docs):
						input_ids, token_type_ids, attention_mask = self.tokenizer.wrap_batch_bert(batch_queries, [bd[i] for bd in batch_docs], padding=True, 
						truncation='only_second', max_length_q=self.max_length_query, max_length_doc=self.max_length_doc, return_tensor=True, add_special_tokens=False)
						encoded_dict = TensorDict({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
						features['encoded_input'].append(encoded_dict) 

				elif self.model == 'rank':
					q_encoded, lengths_q = self.tokenizer.wrap_batch(batch_queries, padding=True,  return_tensor=True, max_length=self.max_length_query, if_empty='<pad>')
					for i in range(self.num_docs):
						d_encoded, lengths_d = self.tokenizer.wrap_batch([bd[i] for bd in batch_docs],  return_tensor=True, padding=True, max_length=self.max_length_doc, if_empty='<pad>')
						#encoded_dict = TensorDict({'X': torch.cat( (q_encoded, d_encoded), 1) , 'lengths_q':  torch.LongTensor(batch_q_lengths), 'lengths_d': torch.LongTensor(batch_d_lengths[:, i])})	
						encoded_dict = TensorDict({'encoded_query': q_encoded, 'encoded_doc': d_encoded , 'lengths_q':  lengths_q, 'lengths_d': lengths_d})	
						features['encoded_input'].append(encoded_dict)


			#print(f'tok_docs time: {time.time() - t_tok_doc}.')	
			#print(f'get_batch time: {time.time() - t_start}.')
					
			yield features
	def collate_fn(self, batch):
		return batch
