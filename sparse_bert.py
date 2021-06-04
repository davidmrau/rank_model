import torch

from torch import nn
from transformers import BertModel, BertConfig


class SparseBERT(nn.Module):

	def __init__(self, sparse_dim, dropout_p, freeze_bert=False):
		super(SparseBERT, self).__init__()
		self.sparse_dim = sparse_dim
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.relu = nn.ReLU()
		self.sparse = nn.Linear(768, sparse_dim)
		self.drop = nn.Dropout(p=dropout_p)
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False
		self.conv = nn.Conv1d(1, sparse_dim, kernel_size=768, stride=768)
		#self.avg = nn.AvgPool1d(sparse_dim, count_include_pad=False)
	def forward(self, input_ids, attention_mask, token_type_ids):
		lengths = attention_mask.sum(1)
		last_hidden_state = self.bert( input_ids, attention_mask, token_type_ids ).last_hidden_state #b x len x hidden
		#return last_hidden_state[:,0, :]
		# replace with proper average pooling
		
		# conv padding not yet handled correctly
		cat_hidden = last_hidden_state.view(-1, last_hidden_state.shape[1] * last_hidden_state.shape[2]).unsqueeze(1)
		#print(cat_hidden.shape)
		out = self.conv(cat_hidden)
		out = self.relu(out)
		#print(out.shape)
		#out = out.mean(2)

		out = out.permute(0,2,1)

		target_shape = [last_hidden_state.shape[0], last_hidden_state.shape[1], self.sparse_dim]
		mask = torch.zeros(target_shape)
		range_tensor = torch.arange(target_shape[1]).unsqueeze(0)
		if out.is_cuda:
			range_tensor = range_tensor.cuda()

		range_tensor = range_tensor.expand(lengths.size(0), range_tensor.size(1))
		mask = (range_tensor < lengths.unsqueeze(1))
		average = (mask.unsqueeze(2) * out).sum(dim = 1) / lengths.unsqueeze(1)
		
		return average	

		#print(out.shape)
		out = self.relu(out)
		return out

		# self implemented version
		packed_hidden = last_hidden_state.view(-1, 768) # b x len * hidden
		packed_hidden = self.relu(packed_hidden)
		packed_hidden = self.drop(packed_hidden)
		
		sparse_hidden = self.sparse(packed_hidden) # b x len * hidden
		sparse_hidden = self.relu(sparse_hidden)
		target_shape = [last_hidden_state.shape[0], last_hidden_state.shape[1], self.sparse_dim]
		sparse_hidden = sparse_hidden.view(target_shape)
		mask = torch.zeros(target_shape)
		range_tensor = torch.arange(target_shape[1]).unsqueeze(0)
		if sparse_hidden.is_cuda:
			range_tensor = range_tensor.cuda()

		range_tensor = range_tensor.expand(lengths.size(0), range_tensor.size(1))
		mask = (range_tensor <= lengths.unsqueeze(1))
		average = (mask.unsqueeze(2) * sparse_hidden).sum(dim = 1)
		
		return average	
		
