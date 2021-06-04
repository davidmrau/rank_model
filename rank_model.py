import numpy as np
import torch
import torch.nn as nn
import pickle

from transformers import BertModel, BertConfig

class RankModel(nn.Module):

	def __init__(self, hidden_sizes, dropout_p, compositionality_weights, freeze_embeddings=False, bert_layers='all', embedding=None, contextualizer=None):
		super(RankModel, self).__init__()
		
		self.hidden_sizes = hidden_sizes
		out_size = 1
		self.contextualizer = contextualizer
		print( embedding, contextualizer)
		if embedding is not None and contextualizer == 'average':
			# load or randomly initialize embeddings according to parameters
			self.embedding = embedding
			self.embedding_dim = embedding.embedding_dim * 2
			self.vocab_size = embedding.num_embeddings
			self.weighted_average = EmbeddingWeightedAverage(weights = compositionality_weights, vocab_size = self.vocab_size, trainable = True) # (weights, vocab_size, trainable = True)
		elif contextualizer == 'bert':
			if bert_layers == 'all':
				bert_config = None
			else:
				bert_config = BertConfig(num_hidden_layers=int(bert_layers))
			self.embedding = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
			self.embedding_dim = self.embedding.embeddings.word_embeddings.embedding_dim
		else:
			raise NotImplementedError()

		if freeze_embeddings:
			for param in self.embedding.parameters():
				param.requires_grad = False

		else:
			for param in self.embedding.parameters():
				param.requires_grad = True



		# create module list
		self.layers = nn.ModuleList()
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=self.embedding_dim, out_features=hidden_sizes[0]))
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Dropout(p=dropout_p))

		for k in range(len(hidden_sizes)-1):
			self.layers.append(nn.Linear(in_features=hidden_sizes[k], out_features=hidden_sizes[k+1]))
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Dropout(p=dropout_p))
		self.embedding_type = embedding

		#self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=hidden_sizes[-1], out_features=out_size))
		else:
			self.layers.append( nn.Linear(in_features=self.embedding_dim, out_features=out_size))
		self.layers.append(nn.Sigmoid())

	def forward_average(self, encoded_query, encoded_doc, lengths_q, lengths_d):
		# get embeddings of all inps
		emb_q = self.embedding(encoded_query)
		emb_d = self.embedding(encoded_doc)
		# calculate weighted average embedding for all inps
		w_av_q = self.weighted_average(encoded_query, emb_q, lengths = lengths_q)
		w_av_d = self.weighted_average(encoded_doc, emb_d, lengths = lengths_d)
			
		q_d =  torch.cat([w_av_q, w_av_d], dim=1)
		return q_d

	def forward_bert(self, inp):
		#outputs = self.bert(input_ids=inp['input_ids'], token_type_ids=inp['token_type_ids'], attention_mask=inp['attention_mask'])
		outputs = self.embedding(**inp)
		encoded_layers = outputs.last_hidden_state[:,0,:]
		return encoded_layers

	def forward(self, inp):
		if self.contextualizer == 'bert':
			q_d = self.forward_bert(inp)
		else:
			q_d = self.forward_average(**inp)
		# getting scores of joint q_d representation
		for layer in self.layers:
			q_d = layer(q_d)
		return q_d

class EmbeddingWeightedAverage(nn.Module):
	def __init__(self, weights, vocab_size, trainable = True):
		"""
		weights : uniform / random /
				  path_to_file (pickle in the form of tensor.Size(V x 1))
		vocab_size: vocabulary size
		"""
		super(EmbeddingWeightedAverage, self).__init__()

		self.weights = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = 1)
		self.softmax = nn.Softmax(1)
		if weights == "uniform":
			self.weights.weight = torch.nn.Parameter(torch.ones(vocab_size,1), requires_grad=True)
			# pass
		elif weights == "random":
			pass
		# otherwise it has to be a path of a pickle file with the weights in a pytorch tensor form
		else:
			try:
				weight_values = pickle.load(open(weights, 'rb'))
				self.weights.weight = torch.nn.Parameter(torch.from_numpy(weight_values).float().unsqueeze(-1))
			except:
				raise IOError(f'(EmbeddingWeightedAverage) Loading weights from pickle file: {weights} not accessible!')

		if trainable == False:
			self.weights.weight.requires_grad = False
		else:
			self.weights.weight.requires_grad = True




	def forward(self, inp, values, lengths = None, max_length = None, mask = None):
		"""
		inp shape : Bsz x L
		values shape  : Bsz x L x hidden
		lengths shape : Bsz x 1
		mask: if provided, are of shape Bsx x L. Binary mask version of lenghts
		"""
		#return values.sum(1).div(lengths.float().view(-1, 1))

		# calculate the weight of each term
		weights = self.weights(inp)	
		if mask is None:
			if lengths is None:
				raise ValueError("EmbeddingWeightedAverage : weighted_average(), mask and lengths cannot be None at the same time!")

			mask = torch.zeros_like(inp, dtype=torch.bool)
			range_tensor = torch.arange(inp.shape[1]).unsqueeze(0)

			if values.is_cuda:
				range_tensor = range_tensor.cuda()

			range_tensor = range_tensor.expand(lengths.size(0), range_tensor.size(1))
			mask = (range_tensor >=  lengths.unsqueeze(1)) 
		# normalize the weights
		weights = self.softmax(weights.masked_fill(mask.unsqueeze(-1), float('-inf')))
		# we first calculate the weighted sum
		weighted_average = (weights * values ).sum(dim = 1)

		return weighted_average


def load_glove_embedding_weights(DATA_EMBEDDINGS, token2id, embedding_dim=300):
	embedding_dim = len(token2id)
	embedding_weighs = torch.nn.Embedding(vocab_size, embedding_dim) 
	glove = {}
	# read glove
	with open( DATA_EMBEDDINGS, 'r') as f:
		for line in f:
			term, weights = line.split('t')
			weights_np = np.from_string(weights, dtype=int, sep=' ')
			glove[term] = weights_np


	for token in token2id:
		if token in glove:
			embedding_weights[token2id[token]] = glove[token]
	return embedding_weights
