import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


import os.path
import datetime
import numpy as np
import shutil
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import seaborn as sns
from matplotlib import pyplot as plt
from file_interface import File
from metrics import MAPTrec
from utils import load_glove_embedding, l2_reg, load_bert_embedding, load_rand_embedding
from data_reader import DataReader
import argparse


from transformers import AutoModelForSequenceClassification


#import csv

parser = argparse.ArgumentParser()
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--model", type=str, default='sparseBERT')
parser.add_argument("--encoded", action='store_true')
parser.add_argument("--mb_size", type=int, default=1024)
parser.add_argument("--single_gpu", action='store_true')
args = parser.parse_args()
print(args)
model_type = 'sparseBERT'
encoded=args.encoded
if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")

MAX_QUERY_TERMS = None
MAX_DOC_TERMS = 512
MB_SIZE = args.mb_size
encoding = 'bert' #  'glove' or 'bert'

#experiments_path = 'project/draugpu/experiments_rank_model/'
MODEL_DIR = "/".join(args.model.split('/')[:-1])
MODEL_DIR += args.add_to_dir



QRELS_TEST = "data/msmarco/2020-qrels-pass-no1.txt"
# TODO: this has to be changed and is only for a first experiment
DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_54.tsv"

ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv"
ID2DOC = 'data/msmarco/collection.tsv'



# load id2q 
id2q_test = File(ID2Q_TEST, encoded=args.encoded)
id2d = File(ID2DOC, encoded=args.encoded)



# instantiate Data Reader
dataset_test = DataReader(model_type, DATA_FILE_TEST, 1, False, id2q_test, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=None, encoded=encoded, sample_random_docs=False, encoding=encoding)
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_test.collate_fn)
# instanitae model

#model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
model = torch.load(args.model)
model = model.module
sparse_dim = model.sparse.weight.shape[0]
def print_message(s):
	print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

print_message('Starting')

model = model.to(DEVICE)
if torch.cuda.device_count() > 1 and not args.single_gpu:
	model = torch.nn.DataParallel(model)
model.activation = 'relu'
model.eval()
res_test = {}
original_scores = {}


latent_words_query = torch.zeros(sparse_dim).to(DEVICE)
latent_words_doc = torch.zeros(sparse_dim).to(DEVICE)
last_query = None
with torch.no_grad():
	for num_i, features in enumerate(dataloader_test):
		if model_type == 'rank':
			# forward doc
			out = model(features['encoded_input'][0].to(DEVICE))
			# to cpu
		elif model_type == 'sparseBERT':

			out_queries = model(**features['encoded_queries'].to(DEVICE))
			out_docs = model(**features['encoded_docs'][0].to(DEVICE)) 

			# how often each latent word was used
			latent_words_query += (out_queries[0] != 0).float()
			latent_words_doc += (out_docs != 0).float().mean(0)
			# l0 loss
			l0_query = (out_queries == 0).float().mean(1).mean(0)
			l0_docs = (out_docs == 0).float().mean(1).mean(0)

			if last_query == None:
				last_query = latent_words_query[0]
			
			if last_query != latent_words_query[0]:
				last_query = latent_words_query[0]

			print('l0 query', l0_query) 
			print('l0 docs', l0_docs) 
			print('unused latent words / query', (latent_words_query == 0).sum())	
			print('unused latent words / doc', (latent_words_doc == 0).sum())	
			out = torch.bmm(out_queries.unsqueeze(1), out_docs.unsqueeze(-1)).squeeze()
		out = out.data.cpu()
		batch_num_examples = len(features['meta'])
		# for each example in batch
		for i in range(batch_num_examples):
			q = features['meta'][i][0]
			d = features['meta'][i][1]
			# sanity check store orcale scores as well
			#orig_score = features['meta'][i][2]
			if q not in res_test:
				res_test[q] = {}
				original_scores[q] = {}
			if d not in res_test[q]:
				res_test[q][d] = 0
		#		original_scores[q][d] = 0
#		original_scores[q][d] += orig_score
			#res_test[q][d] += out[i][0].detach().numpy()
			res_test[q][d] += out[i].detach().numpy()
		# if number of examples < batch size we are done


sorted_scores = []
#sorted_scores_original = []
q_ids = []
# for each query sort after scores
for qid, docs in res_test.items():
	sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
	q_ids.append(qid)
	sorted_scores.append(sorted_scores_q)

plt.figure()
plt.plot(np.arange(latent_words_query.shape[0]), np.sort(latent_words_query.cpu().numpy()) )
#sns.distplot(-np.sort(latent_words_query.cpu().numpy()), bins=latent_words_query.shape[0]//10)
plt.savefig(f'{MODEL_DIR}/latent_words_query', bbox_inches='tight')


plt.figure()
plt.plot( np.arange(latent_words_doc.shape[0]), np.sort(latent_words_doc.cpu().numpy())) 
#sns.distplot(-np.sort(latent_words_doc.cpu().numpy()), bins=latent_words_doc.shape[0]//10)
plt.savefig(f'{MODEL_DIR}/latent_words_doc', bbox_inches='tight')
