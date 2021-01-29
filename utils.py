import numpy as np
import torch
import nltk
import pickle
from torch.nn.utils.rnn import pad_sequence

class BasicTokenizer():

	def __init__(self, token2id):
		self.token2id = token2id

	def __call__(self, batch_texts, return_tensor=False, padding=False, max_length=None, if_empty=None):
		batch_encoded_ids = list()
		batch_lengths = list()
		batch_max_length = 0
		for text in batch_texts:
			tokenized = nltk.word_tokenize(text)
			encoded_ids = list()
			for token in tokenized:
				try:
					id_ = self.token2id[token]
				except:
					id_ = self.token2id['<unk>']
				encoded_ids.append(id_)

			encoded_ids = encoded_ids[:max_length]
			if len(encoded_ids) == 0:
				if if_empty != None:
					encoded_ids = [self.token2id[if_empty]]	
			if len(encoded_ids) > batch_max_length:
				batch_max_length = len(encoded_ids)
			batch_lengths.append(len(encoded_ids))
			batch_encoded_ids.append(encoded_ids)
			
		if padding:
			batch_encoded_ids_pad = np.zeros((len(batch_lengths), max_length if max_length else max(batch_lengths)))
			for i, ids in enumerate(batch_encoded_ids):
				batch_encoded_ids_pad[ i , :len(ids)] = ids
			batch_encoded_ids = batch_encoded_ids_pad
	
		if return_tensor:
			batch_encoded_ids = torch.LongTensor(batch_encoded_ids)

		return batch_encoded_ids, batch_lengths	
			
 
def load_glove_embedding_weights(DATA_EMBEDDINGS, token2id, embedding_dim=300):
	vocab_size = len(token2id)
	embedding_weights = torch.nn.Embedding(vocab_size, embedding_dim) 
	glove = {}
	# read glove
	with open( DATA_EMBEDDINGS, 'r') as f:
		for line in f:

			delim_pos = line.find('\t')
			term, weights = line[:delim_pos], line[delim_pos+1:]
			weights_np = np.fromstring(weights, dtype=int, sep=' ')
			glove[term] = weights_np

	num_not_found = 0
	for token in token2id:
		if token in glove:
			embedding_weights[token2id[token]] = glove[token]
		else:
			num_not_found += 1
	print(f'Done loading glove embeddings. {num_not_found} tokens were not found in the glove vocabulary.')
	return embedding_weights


class TensorDict():
	def __init__(self, d):
		self.d = d
	def to(self, DEVICE):
		for k in self.d:
			if isinstance(self.d[k], torch.Tensor):
				self.d[k] = self.d[k].to(DEVICE)	
		return self.d
