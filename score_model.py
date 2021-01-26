import numpy as np
import torch
import torch.nn as nn
import pickle



class ScoreModel(nn.Module):

	def __init__(self, hidden_sizes, embedding_parameters, embedding_dim, vocab_size, dropout_p, weights, trainable_weights):
		super(ScoreModel, self).__init__()

		self.model_type = 'score-interaction'

		self.hidden_sizes = hidden_sizes

		# load or randomly initialize embeddings according to parameters
		if embedding_parameters is None:
			self.embedding_dim = embedding_dim
			self.vocab_size = vocab_size
			self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
			# set embeddings for model
		else:
			self.embedding = nn.Embedding.from_pretrained(embedding_parameters, freeze=False)
			self.embedding_dim = embedding_parameters.size(1)
			self.vocab_size = embedding_parameters.size(0)

		self.weighted_average = EmbeddingWeightedAverage(weights = weights, vocab_size = self.vocab_size, trainable = trainable_weights) # (weights, vocab_size, trainable = True)

		out_size = 1

		# create module list
		self.layers = nn.ModuleList()
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=self.embedding_dim * 2, out_features=hidden_sizes[0]))
			self.layers.append(nn.Dropout(p=dropout_p))
			self.layers.append(nn.ReLU())

		for k in range(len(hidden_sizes)-1):
			self.layers.append(nn.Linear(in_features=hidden_sizes[k], out_features=hidden_sizes[k+1]))
			self.layers.append(nn.Dropout(p=dropout_p))
			self.layers.append(nn.ReLU())

		#self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=hidden_sizes[-1], out_features=out_size))
		else:
			self.layers.append( nn.Linear(in_features=embedding_dim * 2, out_features=out_size))
		print(self)

	def forward(self, q, doc, lengths_q=None, lengths_d=None):

		# get embeddings of all inps
		emb_q = self.embedding(q)
		emb_d = self.embedding(doc)
		# calculate weighted average embedding for all inps
		w_av_q = self.weighted_average(q, emb_q, lengths = lengths_q)
		w_av_d = self.weighted_average(doc, emb_d, lengths = lengths_d)
		q_d =  torch.cat([w_av_q, w_av_d], dim=1)
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




	def forward(self, inp, values, lengths = None, mask = None):
		"""
		inp shape : Bsz x L
		values shape  : Bsz x L x hidden
		lengths shape : Bsz x 1
		mask: if provided, are of shape Bsx x L. Binary mask version of lenghts
		"""
		if mask is None:

			if lengths is None:
				raise ValueError("EmbeddingWeightedAverage : weighted_average(), mask and lengths cannot be None at the same time!")

			mask = torch.zeros_like(inp)

			for i in range(lengths.size(0)):
				mask[i, : lengths[i].int()] = 1

			if values.is_cuda:
				mask = mask.cuda()

		mask = mask.unsqueeze(-1).float()
		# calculate the weight of each term
		weights = self.weights(inp)
		# normalize the weights
		weights = torch.nn.functional.softmax(weights.masked_fill((1 - mask).bool(), float('-inf')), dim=1)

		# weights are extended to fit the size of the embeddings / hidden representation
		weights = weights.repeat(1,1,values.size(-1))
		# mask are making sure that we only add the non padded tokens
		mask = mask.repeat(1,1,values.size(-1))
		# we first calculate the weighted sum
		weighted_average = (weights * values * mask).sum(dim = 1)
		return weighted_average
