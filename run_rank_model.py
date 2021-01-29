from __future__ import print_function
import os.path
import datetime
import numpy as np
import shutil
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from file_interface import File
from metrics import MAPTrec
from rank_model import RankModel
from utils import load_glove_embedding_weights
from data_reader import DataReader

MODEL='score'
DEVICE = torch.device("cuda")  # torch.device("cpu"), if you want to run on CPU instead
MAX_QUERY_TERMS = 20
MAX_DOC_TERMS = 1500
NUM_HIDDEN_NODES = 300
MB_SIZE = 256
EPOCH_SIZE = 150
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
test_every = 1

L2_LAMBDA = 0.000005
DATA_DIR = 'data'
MODEL_DIR = 'experiments_score/bert_rank_model'
DATA_FILE_VOCAB = os.path.join(DATA_DIR, "embeddings/word-vocab-small.tsv")
DATA_EMBEDDINGS = os.path.join(DATA_DIR, "glove.6B.{}d.txt".format(NUM_HIDDEN_NODES))
DATA_FILE_IDFS = os.path.join(DATA_DIR, "embeddings/idfnew.norm.tsv")
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "qrels.robust2004.txt_paper_binary_rand_docs_0")
DATA_FILE_TEST = os.path.join(DATA_DIR, "robust04_anserini_TREC_test_top_2000_bm25_fold_test_0")
QRELS_TEST = os.path.join(DATA_DIR, "qrels.robust2004.txt")
MODEL_FILE = os.path.join(MODEL_DIR, "duet.ep{}.dnn")
TOKEN2ID_PATH = os.path.join(DATA_DIR, "robust04_word_frequency.tsv_64000_t2i.p")
token2id = pickle.load(open(TOKEN2ID_PATH, 'rb'))
encoding = 'glove' #  'glove' or 'bert'
#compostionality_weights = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
compostionality_weights = 'uniform'



id2q = File('data/trec45.tsv', tokenized=False)
id2d = File('data/robust04_raw_docs.num_query', tokenized=False)

if os.path.exists(f'{MODEL_DIR}/log/'):
	shutil.rmtree(f'{MODEL_DIR}/log/')

writer = SummaryWriter(f'{MODEL_DIR}/log/')
# instantiate Data Reader
READER_TEST = DataReader(encoding, DATA_FILE_TEST, 1, False, id2q, id2d, MB_SIZE, max_length=1500, token2id=token2id)
READER_TRAIN = DataReader(encoding, DATA_FILE_TRAIN, 2, True, id2q, id2d, MB_SIZE, max_length=1500, token2id=token2id)

if encoding == 'glove':
	embedding = load_glove_embedding_weights(DATA_EMBEDDINGS, token2id)
else:
	embedding = encoding
# instanitae model
model = RankModel([], embedding, 0.3, compostionality_weights, freeze_embeddings=True)


def print_message(s):
	print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

def l2_reg(model):
	l2 = torch.tensor(0., device=DEVICE)
	for name, param in model.named_parameters():
		if 'bert' not in name and 'embedding' not in name:
	    		l2 += torch.norm(param)
	return l2

print_message('Starting')
print_message('Learning rate: {}'.format(LEARNING_RATE))

#if torch.cuda.device_count() > 1:
#	model = nn.DataParallel(model)
model = model.to(DEVICE)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MarginRankingLoss(margin=1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

total_examples_seen = 0
for ep_idx in range(NUM_EPOCHS):
	# TRAINING
	model.train()
	train_loss = 0.0
	for mb_idx in range(EPOCH_SIZE):
		print(f'MB {mb_idx + 1}/{EPOCH_SIZE}')
		# get train data
		features = READER_TRAIN.get_minibatch()
		# forward pos and neg document and store score in tuple
		out = tuple([model(features['encoded_input'][i].to(DEVICE)) for i in range(READER_TRAIN.num_docs)])
		#out = torch.cat(out, 1)
		#loss = criterion(out, torch.from_numpy(features['labels']).to(DEVICE))
		# margin ranking loss 
		loss = criterion(out[0], out[1], torch.from_numpy(features['labels']).to(DEVICE))
		L2 =  L2_LAMBDA * l2_reg(model)
		#loss +=L2 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item() + L2
		total_examples_seen += out[0].shape[0]
		acc = np.array(((out[0] > out[1]).int()).float().cpu().mean())
		print_message('examples:{}, loss:{}, l2:{}, acc:{}'.format(total_examples_seen, loss, L2,  acc))
		writer.add_scalar('Train/Loss', loss, total_examples_seen)
		writer.add_scalar('Train/l2', L2, total_examples_seen)
		writer.add_scalar('Train/Accuracy', acc, total_examples_seen)

	print_message('epoch:{}, loss:{}'.format(ep_idx + 1, train_loss / (EPOCH_SIZE+1) ))
	torch.save(model, MODEL_FILE.format(ep_idx + 1))

	# TESTING
	res_test = {}
	original_scores = {}
	if ep_idx % test_every == 0:
		is_complete = False
		READER_TEST.reset()
		model.eval()
		while not is_complete:
			# get test data 
			features = READER_TEST.get_minibatch()
			# forward doc
			out = model(features['encoded_input'][0].to(DEVICE))
			# to cpu
			out = out.data.cpu()
			batch_num_examples = len(features['meta'])
			
			# for each example in batch
			for i in range(batch_num_examples):
				q = features['meta'][i][0]
				d = features['meta'][i][1]
				# sanity check store orcale scores as well
				orig_score = features['meta'][i][2]
				if q not in res_test:
					res_test[q] = {}
					original_scores[q] = {}
				if d not in res_test[q]:
					res_test[q][d] = -10000
					original_scores[q][d] = -10000
				original_scores[q][d] = orig_score
				res_test[q][d] += out[i][0].detach().numpy()
			# if number of examples < batch size we are done
			is_complete = (batch_num_examples < MB_SIZE)

		sorted_scores = []
		sorted_scores_original = []
		q_ids = []
		# for each query sort after scores
		for qid, docs in res_test.items():
			sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
			sorted_scores_original_q = [(doc_id, original_scores[qid][doc_id]) for doc_id in sorted(original_scores[qid], key=original_scores[qid].get, reverse=True)]
			q_ids.append(qid)
			sorted_scores.append(sorted_scores_q)
			sorted_scores_original.append(sorted_scores_original_q)


		# RUN TREC_EVAL
		test = MAPTrec('trec_eval', QRELS_TEST, 1000, ranking_file_path=f'{MODEL_DIR}/ranking')
		map_1000 = test.score(sorted_scores, q_ids)

		map_1000_original = test.score(sorted_scores_original, q_ids)
		print_message('original ranking model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		print_message('model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		writer.add_scalar('Test/map@1000', map_1000, total_examples_seen)
