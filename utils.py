import numpy as np
import torch
import nltk
import pickle
from torch.nn.utils.rnn import pad_sequence
import time
import os 
from transformers import BertConfig, BertModel
class BasicTokenizer():

	def __init__(self, token2id):
		self.token2id = token2id

	def __call__(self, batch_texts, return_tensor=False, padding=False, max_length=None, if_empty=None):
		batch_encoded_ids = list()
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
			batch_encoded_ids.append(encoded_ids)

		batch_encoded_ids, batch_lengths = self.wrap_batch(batch_encoded_ids, max_length=max_length, padding=padding, return_tensor=return_tensor, if_empty=if_empty)
		#print(f'encode_time: {time.time() - t_token}')
		return batch_encoded_ids, batch_lengths	


	def pad(self, batch_ids, max_length=None, if_empty=None):
		batch_ids = [enc[:max_length] for enc in batch_ids]
		batch_lengths = [len(enc) for enc in batch_ids]
		batch_max_length = max(batch_lengths)
		ids_pad = np.zeros((len(batch_lengths), min(batch_max_length, max_length) if max_length else batch_max_length))
		for i, ids in enumerate(batch_ids):
			if len(ids) == 0:
				if if_empty != None:
					ids = [self.token2id[if_empty]]	
			ids_pad[ i , :len(ids)] = ids
		return ids_pad, batch_lengths

	
	def wrap_batch_bert(self, q_batch_encoded_ids, d_batch_encoded_ids,  max_length_q=None, max_length_doc=None, padding=False, return_tensor=False, if_empty=None, add_special_tokens=False, truncation='only_second'):

		if truncation != 'only_second':
			raise NotImplementedError()

		if add_special_tokens:
			raise NotImplementedError()

		
		batch_lengths_q = [len(enc_q) + 2 for enc_q in q_batch_encoded_ids]
		
		batch_comb, token_type_ids = list(), list()
		for i in range(len(q_batch_encoded_ids)):	
			q, d, q_len = q_batch_encoded_ids[i], d_batch_encoded_ids[i], batch_lengths_q[i]
			d = d[:max_length_doc - q_len - 1 ]
			
			comb = np.concatenate(( [101], q, [102],  d, [102]))
			pad_arr = (max_length_doc - len(comb)) * [0]
			token_type_id = q_len * [0] + (max_length_doc - q_len) * [1]
			comb_pad = np.concatenate((comb, pad_arr))
			batch_comb.append(comb_pad)
			token_type_ids.append(token_type_id)

		batch_encoded_ids, batch_lengths = self.wrap_batch(batch_comb, padding=False, return_tensor=return_tensor, if_empty=if_empty)
		attention_mask = (batch_encoded_ids != 0).long()
		return batch_encoded_ids, torch.LongTensor(token_type_ids), attention_mask


	def wrap_batch(self, batch_encoded_ids, max_length=None, padding=False, return_tensor=False, if_empty=None):
		#t_pad = time.time()
		if max_length:
			batch_encoded_ids = [ids[:max_length] for ids in batch_encoded_ids]
		if padding:
			batch_encoded_ids, batch_lengths = self.pad(batch_encoded_ids, max_length=max_length, if_empty=if_empty)
		else:
			batch_lengths = [len(enc) for enc in batch_encoded_ids]
		#print(f'pad_time: {time.time() - t_pad}')
		if return_tensor:
			batch_encoded_ids = torch.LongTensor(batch_encoded_ids)
			batch_lengths = torch.LongTensor(batch_lengths)
		return batch_encoded_ids, batch_lengths	

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
		#vocab_size = len(token2id)
		vocab_size = max(token2id.values()) + 1
		weight_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
		print(f'Embeddings size: {vocab_size}, {embedding_dim}')
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

def load_rand_embedding(vocab_size, dim):
	return torch.nn.Embedding(vocab_size, dim)
def load_bert_embedding():
	bert_config = BertConfig(num_hidden_layers=1)
	bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
	embedding = torch.nn.Embedding.from_pretrained(bert.embeddings.word_embeddings.weight, freeze=False)	
	return embedding


class TensorDict():
	def __init__(self, d):
		self.d = d
	def to(self, DEVICE):
		for k in self.d:
			if isinstance(self.d[k], torch.Tensor):
				self.d[k] = self.d[k].to(DEVICE)	
		return self.d

	def __getitem__(self, key):
		return self.d[key]


def l2_reg(model):
	l2 = 0
	for name, param in model.named_parameters():
		if 'bert' not in name and 'embedding' not in name:
	    		l2 += torch.norm(param)
	return l2



class BERT_inter(torch.nn.Module):
	def __init__(self, hidden_size = 256, num_of_layers = 2, num_attention_heads = 4, input_length_limit = 512,
			vocab_size = 30522, embedding_parameters = None, params_to_copy = {}):
		super(BERT_inter, self).__init__()


		self.model_type = "bert-interaction"
		output_size = 2

		if embedding_parameters is not None:
			# adjust hidden size and vocab size
			hidden_size = embedding_parameters.size(1)
			vocab_size = embedding_parameters.size(0)


		intermediate_size = hidden_size*4

		# set up the Bert config
		config = transformers.BertConfig(vocab_size = vocab_size, hidden_size = hidden_size, num_hidden_layers = num_of_layers,
										num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = input_length_limit)

		self.encoder = transformers.BertModel(config)

		self.last_linear = torch.nn.Linear(hidden_size, hidden_size)

		self.output_linear = torch.nn.Linear(hidden_size, output_size)
		# copy all specified parameters
		for param in params_to_copy:

			param_splitted = param.split(".")

			item = self.encoder.__getattr__(param_splitted[0])

			for p in param_splitted[1: -1]:
				item  = item.__getattr__(p)

			last_item = param_splitted[-1]

			setattr(item, last_item, params_to_copy[param])

	def forward(self, input_ids, attention_masks, token_type_ids, output_attentions=False, output_hidden_states=False):
		all_out = self.encoder(input_ids = input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
		last_hidden_state = all_out.last_hidden_state

		# extract hidden representation of the 1st token, that is the CLS special token
		cls_hidden_repr = last_hidden_state[:,0]

		out = self.last_linear(cls_hidden_repr)

		out = torch.tanh(out)
