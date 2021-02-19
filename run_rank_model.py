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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from file_interface import File
from metrics import MAPTrec
from rank_model import RankModel
from utils import load_glove_embedding, l2_reg, load_bert_embedding
from data_reader import DataReader
import argparse

#import csv

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", type=str, default='glove')
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--mb_size", type=int, default=256)
parser.add_argument("--bert_layers", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--l2_lambda", type=float, default=0.001)
parser.add_argument("--encoded", action='store_true')
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--sample_random_docs", action='store_true')
parser.add_argument("--freeze_embeddings", action='store_true')
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--compositionality_weights", type=str, default='uniform')
parser.add_argument('--hidden_dim', type=str, default='128-128')
args = parser.parse_args()
print(args)

encoded=args.encoded
if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")

MAX_QUERY_TERMS = 512
MAX_DOC_TERMS = 512
MB_SIZE = args.mb_size
EPOCH_SIZE = 500
NUM_EPOCHS = 40
LEARNING_RATE = args.lr
test_every = 1
log_every = EPOCH_SIZE // 4
encoding = args.encoding #  'glove' or 'bert'
L2_LAMBDA = args.l2_lambda 

sample_random_docs = '' if not args.sample_random_docs else '_sample_random_docs_'
freeze_embeddings = '' if not args.freeze_embeddings else '_freeze_emb_'
freeze_embeddings = '' if not args.encoded else '_encoded_'
bert_layers = '' if args.encoding == 'glove' and args.bert_layers == None  else f'_bert_layers_{args.bert_layers}_'
MODEL_DIR = f'experiments_{args.dataset}/model_hs_{args.hidden_dim}_{args.encoding}_bz_{args.mb_size}_lr_{args.lr}_l2_{args.l2_lambda}_{args.compositionality_weights}_do_{args.dropout}_{freeze_embeddings}{encoded}{sample_random_docs}'
MODEL_DIR += args.add_to_dir


# other data paths
DATA_EMBEDDINGS = "data/embeddings/glove.6B.300d.txt"
DATA_FILE_IDFS = "data/embeddings/idfnew.norm.tsv"
#TOKEN2ID_PATH = "data/robust/robust04_word_frequency.tsv_64000_t2i.p"
TOKEN2ID_PATH = "data/robust/robust04_word_frequency.tsv_400000_t2i.p"
token2id = pickle.load(open(TOKEN2ID_PATH, 'rb'))


compostionality_weights = args.compositionality_weights

if args.dataset == 'robust':
	#compostionality_weights = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
	DATA_FILE_TRAIN = "data/robust_test/qrels.robust2004.txt_paper_binary_rand_docs_shuf_train"
	QRELS_TEST = "data/robust_test/qrels.robust2004.txt"
	DATA_FILE_TEST = "data/robust_test/robust04_anserini_TREC_test_top_2000_bm25_test"

	if 'glove' in args.encoding and args.encoded and args.bert_layers == None:
		ID2Q_TEST = 'data/robust_test/trec45-t.tsv_glove_stop_lucene_64k.tsv'
		ID2Q_TRAIN = ID2Q_TEST
		# Weak supervision
		#ID2Q_TRAIN = File('data/aol/aol_remove_test.tsv.names_glove_stop_lucene_64k.tsv', encoded=True)
		ID2DOC = 'data/robust/robust04_raw_docs.num_query_glove_stop_lucene_64k.tsv'
	elif 'bert' in args.encoding and args.encoded or args.bert_layers != None:
		ID2Q_TEST = 'data/robust_test/trec45-t.tsv_bert_stop_none.tsv'
		ID2DOC = 'data/robust/robust04_raw_docs.num_query_bert_stop_none.tsv'
		ID2Q_TRAIN = ID2Q_TEST
		# Weak supervision
		# missing 
	else:
		ID2Q_TEST = 'data/robust_test/trec45-t.tsv'
		ID2Q_TRAIN = ID2Q_TEST
		# Weak supervision
		#ID2Q_TRAIN = 'data/aol/aol_remove_test.tsv.names'
		ID2DOC = 'data/robust/robust04_raw_docs.num_query'
elif  args.dataset == 'clueweb':
	#compostionality_weights = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
	DATA_FILE_TRAIN = "data/webtrack/qrels.web.1-200.txt_ranking_results_spam_filtered_paper_binary_shuf_train"
	QRELS_TEST = "data/webtrack/qrels.web.1-200.txt"
	# TODO: this has to be changed and is only for a first experiment
	DATA_FILE_TEST = "data/webtrack/topics.web.1-200.xml.run.cw09b.bm25.top-2000_stemmed_remove_stop_spam_filtered_test"

	if 'glove' in args.encoding and args.encoded and args.bert_layers == None:
		ID2Q_TEST = 'data/webtrack/topics.web.1-200.txt_glove_stop_lucene.tsv' 
		ID2Q_TRAIN = ID2Q_TEST
		# Weak supervision
		#ID2Q_TRAIN =  
		ID2DOC = "data/clueweb/clueweb09b_docs_cleaned_docs_in_comb_spam_filtered_glove_stop_lucene.tsv" 
	elif 'bert' in args.encoding  and args.encoded or args.bert_layers != None:
		raise NotImplementedError()
		#ID2Q_TEST = 
		#ID2Q_TRAIN = 
		# Weak supervision
		# missing 
		#ID2DOC =
	else:
		raise NotImplementedError()
		#ID2Q_TEST = 'data/webtrack/topics.web.1-200.txt'
		#ID2Q_TRAIN = ID2Q_TEST 
		# Weak supervision
		#ID2Q_TRAIN = 
		#ID2DOC = 
elif  args.dataset == 'msmarco':
	#compostionality_weights = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
	DATA_FILE_TRAIN = "data/msmarco/qidpidtriples.train.full.tsv"
	QRELS_TEST = "data/msmarco/2020-qrels-pass-no1.txt"
	# TODO: this has to be changed and is only for a first experiment
	DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style.tsv"

	if 'glove' in args.encoding and args.encoded and args.bert_layers == None:
		raise NotImplementedError()
		#ID2Q_TEST =  
		#ID2Q_TRAIN = ID2Q_TEST
		# Weak supervision
		#ID2Q_TRAIN =  
		#ID2DOC =  
	elif 'bert' in args.encoding  and args.encoded or args.bert_layers != None:
		ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv_bert_stop_none_remove_unk.tsv"
		ID2Q_TRAIN = "data/msmarco/queries.train.tsv_bert_stop_none_remove_unk.tsv" 
		ID2DOC = "data/msmarco/collection.tsv_bert_stop_none_remove_unk.tsv"
	else:
		raise NotImplementedError()
		#ID2Q_TEST = 'data/webtrack/topics.web.1-200.txt'
		#ID2Q_TRAIN = ID2Q_TEST 
		# Weak supervision
		#ID2Q_TRAIN = 
else:
	raise NotImplementedError()
# load id2q 
id2q_test = File(ID2Q_TEST, encoded=args.encoded)
id2q_train = File(ID2Q_TRAIN, encoded=args.encoded)
id2d = File(ID2DOC, encoded=args.encoded)


MODEL_FILE = os.path.join(MODEL_DIR, "rank_model.ep{}.pth")

print(f'Saving model to {MODEL_DIR}')
if os.path.exists(f'{MODEL_DIR}/log/'):
	shutil.rmtree(f'{MODEL_DIR}/log/')

writer = SummaryWriter(f'{MODEL_DIR}/log/')
# instantiate Data Reader
dataset_test = DataReader(encoding, DATA_FILE_TEST, 1, False, id2q_test, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=token2id, encoded=encoded, sample_random_docs=args.sample_random_docs)
dataset_train = DataReader(encoding, DATA_FILE_TRAIN, 2, True, id2q_train, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=token2id, encoded=encoded, sample_random_docs=args.sample_random_docs)
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_test.collate_fn)
dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_train.collate_fn)

if encoding == 'glove' and args.bert_layers == None:
	embedding = load_glove_embedding(DATA_EMBEDDINGS, TOKEN2ID_PATH)
if encoding == 'glove' and args.bert_layers == 'embeddings':
	embedding = load_bert_embedding()
else:
	embedding = encoding


# instanitae model
model = RankModel([ int(h) for h in args.hidden_dim.split('-')], embedding, args.dropout, compostionality_weights, freeze_embeddings=args.freeze_embeddings, bert_layers=args.bert_layers)


def print_message(s):
	print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


print_message('Starting')

model = model.to(DEVICE)
if torch.cuda.device_count() > 1 and not args.single_gpu:
	model = torch.nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
#criterion = nn.MarginRankingLoss(margin=1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


#qrels = {}
#with open(QRELS_TEST, mode='r', encoding="utf-8") as f:
#	reader = csv.reader(f, delimiter=' ')
#	for row in reader:
#		qid                         = row[0]
#		did                         = row[2]
#		if qid not in qrels:
#			qrels[qid]              = []
#			qrels[qid].append(did)





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
		out = torch.cat(out, 1)
		train_loss = criterion(out, features['labels'].long().to(DEVICE) * 0)
		total_examples_seen += out.shape[0]
		acc = np.array(((out[:, 0] > out[:, 1]).int()).float().cpu().mean())
		# margin ranking loss 
		#train_loss = criterion(out[0], out[1], features['labels'].to(DEVICE))
		#total_examples_seen += out[0].shape[0]
		#acc = np.array(((out[0] > out[1]).int()).float().cpu().mean())
		L2 =  L2_LAMBDA * l2_reg(model)
		loss = train_loss + L2
		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if mb_idx % log_every == 0:
			print(f'MB {mb_idx + 1}/{EPOCH_SIZE}')
			print_message('examples:{}, loss:{}, l2:{}, acc:{}'.format(total_examples_seen, loss, L2,  acc))
			writer.add_scalar('Train/Loss', train_loss, total_examples_seen)
			writer.add_scalar('Train/l2', L2, total_examples_seen)
			writer.add_scalar('Train/Accuracy', acc, total_examples_seen)

		mb_idx += 1
	print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, total_loss / (EPOCH_SIZE) ))
#	model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1  else model.state_dict()
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
				#orig_score = features['meta'][i][2]
				if q not in res_test:
					res_test[q] = {}
					original_scores[q] = {}
				if d not in res_test[q]:
					res_test[q][d] = 0
			#		original_scores[q][d] = 0
		#		original_scores[q][d] += orig_score
				res_test[q][d] += out[i][0].detach().numpy()
			# if number of examples < batch size we are done

		sorted_scores = []
		#sorted_scores_original = []
		q_ids = []
		# for each query sort after scores
		for qid, docs in res_test.items():
			sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
			#sorted_scores_original_q = [(doc_id, original_scores[qid][doc_id]) for doc_id in sorted(original_scores[qid], key=original_scores[qid].get, reverse=True)]
			q_ids.append(qid)
			sorted_scores.append(sorted_scores_q)
			#sorted_scores_original.append(sorted_scores_original_q)

		#mrr = 0
		#for qid, docs in res_test.items():
	#		ranked = sorted(docs, key=docs.get, reverse=True)
#			for i in range(min(len(ranked), 10)):
#				if ranked[i] in qrels[qid]:
#					mrr += 1 / (i + 1)
#					break
#		mrr /= len(qrels)
#		print_message('model:{}, mrr:{}'.format(ep_idx + 1, mrr))


		# RUN TREC_EVAL
		test = MAPTrec('trec_eval', QRELS_TEST, 1000, ranking_file_path=f'{MODEL_DIR}/model_{ep_idx+1}_ranking')

		# sanity check original model
		#map_1000_original = test.score(sorted_scores_original, q_ids)
		#print_message('original ranking model:{}, map@1000:{}'.format(ep_idx + 1, map_1000_original))

		map_1000 = test.score(sorted_scores, q_ids)
		print_message('model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		writer.add_scalar('Test/map@1000', map_1000, total_examples_seen)



		

