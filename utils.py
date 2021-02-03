import numpy as np
import torch
import nltk
import pickle
from torch.nn.utils.rnn import pad_sequence
import time
import os 

class BasicTokenizer():

	def __init__(self, token2id):
		self.token2id = token2id

	def __call__(self, batch_texts, return_tensor=False, padding=False, max_length=None, if_empty=None):
		batch_encoded_ids = list()
		batch_lengths = list()
		#t_token = time.time()
		for text in batch_texts:
			tokenized = nltk.word_tokenize(text)
			encoded_ids = list()
			for token in tokenized:
				try:
					id_ = self.token2id[token]
				except:
					id_ = self.token2id['<unk>']
				encoded_ids.append(id_)

			if len(encoded_ids) == 0:
				if if_empty != None:
					encoded_ids = [self.token2id[if_empty]]	
			batch_lengths.append(len(encoded_ids))
			batch_encoded_ids.append(encoded_ids)

		batch_encoded_ids = self.wrap_batch(batch_encoded_ids, batch_lengths, max_length=max_length, padding=padding, return_tensor=return_tensor, if_empty=if_empty)
		#print(f'encode_time: {time.time() - t_token}')

		return batch_encoded_ids, batch_lengths	
			
	def wrap_batch(self, batch_encoded_ids, batch_lengths, max_length=None, padding=False, return_tensor=False, if_empty=None):
		#t_pad = time.time()
		batch_max_length = np.max(batch_lengths)
		if padding:
			batch_encoded_ids_pad = np.zeros((len(batch_lengths), min(batch_max_length, max_length)))
			for i, ids in enumerate(batch_encoded_ids):
				ids = ids[:max_length]
				if len(ids) == 0:
					if if_empty != None:
						encoded_ids = [self.token2id[if_empty]]	
				batch_encoded_ids_pad[ i , :len(ids)] = ids
			batch_encoded_ids = batch_encoded_ids_pad
		#print(f'pad_time: {time.time() - t_pad}')
		if return_tensor:
			batch_encoded_ids = torch.LongTensor(batch_encoded_ids)
		return batch_encoded_ids	

def load_cache(filename):
	if os.path.exists(filename):
		print(f'Loading {filename} from cache.')
		return pickle.load(open(filename, 'rb'))
	return None

def load_glove_embedding(DATA_EMBEDDINGS, token2id_path, embedding_dim=300):
	token2id = pickle.load(open(token2id_path, 'rb'))
	fname = f'{DATA_EMBEDDINGS}_{token2id_path.split("/")[-1]}_cached'
	# if embedding is cached load
	embedding = load_cache(fname)
	if embedding == None:
		# if not cached make embedding
		vocab_size = len(token2id)
		weight_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
		glove = {}
		# read glove
		with open( DATA_EMBEDDINGS, 'r') as f:
			for line in f:
				delim_pos = line.find(' ')
				term, weights = line[:delim_pos], line[delim_pos+1:]
				weights_np = np.fromstring(weights, dtype=float, sep=' ')
				glove[term] = weights_np

		num_not_found = 0
		for token in token2id:
			if token in glove:
				weight_matrix[token2id[token]] = glove[token]
			else:	
				num_not_found += 1

		print(f'Done loading glove embeddings. {num_not_found} tokens were not found in the glove vocabulary.')

		weight_matrix = torch.FloatTensor(weight_matrix)
		embedding = torch.nn.Embedding.from_pretrained(weight_matrix, freeze=False)
		pickle.dump(embedding, open(fname, 'wb'))
	return embedding


class TensorDict():
	def __init__(self, d):
		self.d = d
	def to(self, DEVICE):
		for k in self.d:
			if isinstance(self.d[k], torch.Tensor):
				self.d[k] = self.d[k].to(DEVICE)	
		return self.d




def l2_reg(model):
	l2 = 0
	for name, param in model.named_parameters():
		if 'bert' not in name and 'embedding' not in name:
	    		l2 += torch.norm(param)
	return l2
