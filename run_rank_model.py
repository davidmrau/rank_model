import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf


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
from torch.utils.data import DataLoader

from file_interface import File
from metrics import MAPTrec
from rank_model import RankModel
from utils import load_glove_embedding, l2_reg
from data_reader import DataReader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", type=str, default='glove')
parser.add_argument("--mb_size", type=int, default=256)
parser.add_argument("--num_bert_layers", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--l2_lambda", type=float, default=0.001)
parser.add_argument("--encoded", action='store_true')
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--freeze_embeddings", action='store_true')
parser.add_argument("--compositionality_weights", type=str, default='uniform')
parser.add_argument('--hidden_dim', type=str, default='128-128')
args = parser.parse_args()

print(args)

encoded=args.encoded
DEVICE = torch.device("cuda")  # torch.device("cpu"), if you want to run on CPU instead
MAX_QUERY_TERMS = 50
MAX_DOC_TERMS = 1500
MB_SIZE = args.mb_size
EPOCH_SIZE = 500
NUM_EPOCHS = 10
LEARNING_RATE = args.lr
test_every = 1
log_every = EPOCH_SIZE // 4
encoding = args.encoding #  'glove' or 'bert'
L2_LAMBDA = args.l2_lambda 
DATA_DIR = 'data'
MODEL_DIR = f'experiments/model_hs_{args.hidden_dim}_{args.encoding}_bz_{args.mb_size}_lr_{args.lr}_l2_{args.l2_lambda}_{args.compositionality_weights}_do_{args.dropout}_fr_emb_{args.freeze_embeddings}_enc_{args.encoded}_n_bert_{args.num_bert_layers}'
DATA_EMBEDDINGS = os.path.join(DATA_DIR, "glove.6B.300d.txt")
DATA_FILE_IDFS = os.path.join(DATA_DIR, "embeddings/idfnew.norm.tsv")
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "qrels.robust2004.txt_paper_binary_rand_docs_0")
DATA_FILE_TEST = os.path.join(DATA_DIR, "robust04_anserini_TREC_test_top_2000_bm25")
QRELS_TEST = os.path.join(DATA_DIR, "qrels.robust2004.txt")
MODEL_FILE = os.path.join(MODEL_DIR, "duet.ep{}.dnn")
#TOKEN2ID_PATH = os.path.join(DATA_DIR, "robust04_word_frequency.tsv_64000_t2i.p")
TOKEN2ID_PATH = os.path.join(DATA_DIR, "robust04_word_frequency.tsv_64000_t2i.p")
token2id = pickle.load(open(TOKEN2ID_PATH, 'rb'))
#compostionality_weights = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
compostionality_weights = args.compositionality_weights


if encoded and encoding == 'glove':
	id2q = File('data/trec45-t.tsv_glove_stop_lucene_64k.tsv', encoded=True)
	id2d = File('data/robust04_raw_docs.num_query_glove_stop_lucene_64k.tsv', encoded=True)
elif encoded and encoding == 'bert':

	id2q = File('data/trec45-t.tsv_bert_stop_none.tsv', encoded=True)
	id2d = File('data/robust04_raw_docs.num_query_bert_stop_none.tsv', encoded=True)
else:
	id2q = File('data/trec45-t.tsv', encoded=False)
	id2d = File('data/robust04_raw_docs.num_query', encoded=False)


print(f'Saving model to {MODEL_DIR}')
if os.path.exists(f'{MODEL_DIR}/log/'):
	shutil.rmtree(f'{MODEL_DIR}/log/')

writer = SummaryWriter(f'{MODEL_DIR}/log/')
# instantiate Data Reader
dataset_test = DataReader(encoding, DATA_FILE_TEST, 1, False, id2q, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=token2id, encoded=encoded)
dataset_train = DataReader(encoding, DATA_FILE_TRAIN, 2, True, id2q, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=token2id, encoded=encoded)
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_test.collate_fn)
dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_train.collate_fn)
if encoding == 'glove':
	embedding = load_glove_embedding(DATA_EMBEDDINGS, TOKEN2ID_PATH)
else:
	embedding = encoding
# instanitae model
model = RankModel([ int(h) for h in args.hidden_dim.split('-')], embedding, args.dropout, compostionality_weights, freeze_embeddings=args.freeze_embeddings, num_bert_layers=args.num_bert_layers)


def print_message(s):
	print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


print_message('Starting')

#if torch.cuda.device_count() > 1:
#	model = nn.DataParallel(model)
model = model.to(DEVICE)
if torch.cuda.device_count() > 1 and not args.single_gpu:
	model = torch.nn.DataParallel(model)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MarginRankingLoss(margin=1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

total_examples_seen = 0

batch_iterator = iter(dataloader_train)
for ep_idx in range(NUM_EPOCHS):
	# TRAINING
	model.train()
	total_loss = 0.0
	mb_idx = 0
	while mb_idx + 1 <   EPOCH_SIZE:
		
		# get train data
		try:
			features = next(batch_iterator)
		except StopIteration:
			batch_iterator = iter(dataloader_train)
			continue
		
		out = tuple([model(features['encoded_input'][i].to(DEVICE)) for i in range(dataset_train.num_docs)])
		#out = torch.cat(out, 1)
		#loss = criterion(out, torch.from_numpy(features['labels']).to(DEVICE))
		# margin ranking loss 
		train_loss = criterion(out[0], out[1], features['labels'].to(DEVICE))
		L2 =  L2_LAMBDA * l2_reg(model)
		loss = train_loss + L2
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_examples_seen += out[0].shape[0]
		acc = np.array(((out[0] > out[1]).int()).float().cpu().mean())
		
		if mb_idx % log_every == 0:
			print(f'MB {mb_idx + 1}/{EPOCH_SIZE}')
			print_message('examples:{}, loss:{}, l2:{}, acc:{}'.format(total_examples_seen, loss, L2,  acc))
			writer.add_scalar('Train/Loss', train_loss, total_examples_seen)
			writer.add_scalar('Train/l2', L2, total_examples_seen)
			writer.add_scalar('Train/Accuracy', acc, total_examples_seen)

		mb_idx += 1
	print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, total_loss / (EPOCH_SIZE) ))
	torch.save(model, MODEL_FILE.format(ep_idx + 1))

	# TESTING
	res_test = {}
	original_scores = {}
	if ep_idx % test_every == 0:
		model.eval()
		for features in dataloader_test:
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
					res_test[q][d] = 100
					original_scores[q][d] = 100
				original_scores[q][d] += orig_score
				res_test[q][d] = out[i][0].detach().numpy()
			# if number of examples < batch size we are done

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
		test = MAPTrec('trec_eval', QRELS_TEST, 1000, ranking_file_path=f'{MODEL_DIR}/model_{ep_idx+1}_ranking')
		map_1000 = test.score(sorted_scores, q_ids)

		#map_1000_original = test.score(sorted_scores_original, q_ids)
		#print_message('original ranking model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		print_message('model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		writer.add_scalar('Test/map@1000', map_1000, total_examples_seen)
