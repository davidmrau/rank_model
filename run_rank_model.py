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
from sparse_bert import SparseBERT
from utils import load_glove_embedding, load_bert_embedding, load_rand_embedding
from data_reader import DataReader
import argparse

#import csv

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", type=str, default='bert')
parser.add_argument("--add_to_dir", type=str, default='')
parser.add_argument("--model", type=str, default='rank')
parser.add_argument("--mb_size", type=int, default=256)
parser.add_argument("--epoch_size", type=int, default=256)
parser.add_argument("--sparse_dim", type=int, default=1000)
parser.add_argument("--bert_layers", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--l1_scalar", type=float, default=0.00001)
parser.add_argument("--encoded", action='store_true')
parser.add_argument("--single_gpu", action='store_true')
parser.add_argument("--sample_random_docs", action='store_true')
parser.add_argument("--freeze_bert", action='store_true')
parser.add_argument("--freeze_embeddings", action='store_true')
parser.add_argument("--random_embeddings", action='store_true')
parser.add_argument("--eval", action='store_true')
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--compositionality_weights", type=str, default='uniform')
parser.add_argument('--hidden_dim', type=str, default='512-512')
args = parser.parse_args()
print(args)


encoded=args.encoded


MAX_QUERY_TERMS = None
MAX_DOC_TERMS = 512
MB_SIZE = args.mb_size
EPOCH_SIZE = args.epoch_size
NUM_EPOCHS = 40
LEARNING_RATE = args.lr
test_every = 1
#log_every = EPOCH_SIZE // 8 if EPOCH_SIZE > 8 else EPOCH_SIZE
log_every = 1

sample_random_docs = '' if not args.sample_random_docs else '_sample_random_docs_'
freeze_embeddings = '' if not args.freeze_embeddings else '_fr_emb_'
random_embeddings = '' if not args.random_embeddings else '_rand_emb_'
encoded = '' if not args.encoded else '_encoded_'
bert_layers = '' if args.bert_layers == None  else f'_bert_layers_{args.bert_layers}_'


experiments_path = f'experiments_{args.model}_model/'

MODEL_DIR = f'{experiments_path}experiments_{args.dataset}/model_bz_{args.mb_size}_lr_{args.lr}_do_{args.dropout}_sr_{sample_random_docs}_sd_{args.sparse_dim}_l1_{str(args.l1_scalar).replace(".", "_")}'
MODEL_DIR += args.add_to_dir


# other data paths
DATA_EMBEDDINGS = "data/embeddings/glove.6B.300d.txt"
DATA_FILE_IDFS = "data/embeddings/idfnew.norm.tsv"
TOKEN2ID_PATH = "data/robust/robust04_word_frequency.tsv_64000_t2i.p"
#TOKEN2ID_PATH = "data/robust/robust04_word_frequency.tsv_400000_t2i.p"
#TOKEN2ID_PATH = 'data/msmarco/collection.tsv.ascii_vocab_count_64000_t2i.p'
#token2id = pickle.load(open(TOKEN2ID_PATH, 'rb'))
token2id = None


compostionality_weights = args.compositionality_weights

if args.dataset == 'robust':
	#compostionality_weights = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
	#DATA_FILE_TRAIN = "data/robust_test/qrels.robust2004.txt_paper_binary_rand_docs_shuf_train"
	DATA_FILE_TRAIN = "data/aol/robust04_AOL_anserini_top_1000_bm25_10k_TRIPLETS_1000_shuf_train"
	QRELS_TEST = "data/robust_test/qrels.robust2004.txt"
	#DATA_FILE_TEST = "data/robust_test/robust04_anserini_TREC_test_top_2000_bm25_test"
	DATA_FILE_TEST = "data/robust_test/robust04_anserini_TREC_test_top_2000_bm25"

	if 'glove' in args.encoding and args.encoded and args.bert_layers == None:
		ID2Q_TEST = 'data/robust_test/trec45-t.tsv_glove_stop_lucene_64k.tsv'
		# Strong Supervision
		#ID2Q_TRAIN = ID2Q_TEST
		# Weak supervision
		ID2Q_TRAIN = 'data/aol/aol_remove_test.tsv.names_glove_stop_lucene_64k.tsv'
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
	#DATA_FILE_TRAIN = "data/msmarco/2020-qrels-pass-no1.txt_paper_binary_train"
	#DATA_FILE_TRAIN = "data/msmarco/2020-qrels-pass-no1.txt_pairwise_train"
	#DATA_FILE_TRAIN = "data/msmarco/2020-qrels-pass-no1.txt_pairwise_binary_train"
	#QRELS_TEST = "data/msmarco/2019qrels-pass.txt"
	QRELS_TEST = "data/msmarco/2020-qrels-pass-no1.txt"
	# TODO: this has to be changed and is only for a first experiment
	#DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv"
	DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_54.tsv"
	#DATA_FILE_TEST = "data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style.tsv_test"

	if 'glove' in args.encoding and args.encoded and args.bert_layers == None:
		ID2Q_TEST =  "data/msmarco/msmarco-test2020-queries.tsv_glove_stop_lucene_64k.tsv"
		#ID2Q_TRAIN = "data/msmarco/queries.train.tsv_glove_stop_lucene_64k.tsv"
		ID2Q_TRAIN = ID2Q_TEST
		ID2DOC =  "data/msmarco/collection.tsv_glove_stop_lucene_64k.tsv"
	#elif 'bert' in args.encoding  and args.encoded or args.bert_layers != None:
	#	ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv_bert_stop_none_remove_unk.tsv"
	#	ID2Q_TRAIN = ID2Q_TEST
	#	#ID2Q_TRAIN = "data/msmarco/queries.train.tsv_bert_stop_none_remove_unk.tsv" 
	#	ID2DOC = "data/msmarco/collection.tsv_bert_stop_none_remove_unk.tsv"
	else:
		ID2Q_TEST = "data/msmarco/msmarco-test2020-queries.tsv"
		#ID2Q_TEST = "data/msmarco/msmarco-test2019-queries_43.tsv"
		ID2Q_TRAIN = "data/msmarco/queries.train.tsv" 
		ID2DOC = "data/msmarco/collection.tsv"
else:
	raise NotImplementedError()
# load id2q 
id2q_test = File(ID2Q_TEST, encoded=args.encoded)
id2q_train = File(ID2Q_TRAIN, encoded=args.encoded)
id2d = File(ID2DOC, encoded=args.encoded)


MODEL_FILE = os.path.join(MODEL_DIR, "rank_model.ep{}.pth")

print(f'Saving model to {MODEL_DIR}')
if os.path.exists(f'{MODEL_DIR}/log/'):
		user_inp = input(f"Type 'y' to delete folder: {MODEL_DIR}")
		if user_inp == 'y':
			shutil.rmtree(f'{MODEL_DIR}/log/')
		else:
			exit()

writer = SummaryWriter(f'{MODEL_DIR}/log/')
# instantiate Data Reader
dataset_test = DataReader(args.model, DATA_FILE_TEST, 1, False, id2q_test, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=token2id, encoded=encoded, sample_random_docs=args.sample_random_docs, encoding=args.encoding)
dataset_train = DataReader(args.model, DATA_FILE_TRAIN, 2, True, id2q_train, id2d, MB_SIZE, max_length_doc=MAX_DOC_TERMS, max_length_query=MAX_QUERY_TERMS, token2id=token2id, encoded=encoded, sample_random_docs=args.sample_random_docs, encoding=args.encoding)
dataloader_test = DataLoader(dataset_test, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_test.collate_fn)
dataloader_train = DataLoader(dataset_train, batch_size=None, num_workers=1, pin_memory=True, collate_fn=dataset_train.collate_fn)
if args.encoding == 'glove' and args.random_embeddings:
	embedding = load_rand_embedding(64000, 300)
	contextualizer = 'average'
elif args.encoding == 'glove' and args.bert_layers == None:
	embedding = load_glove_embedding(DATA_EMBEDDINGS, TOKEN2ID_PATH)
	contextualizer = 'average'
elif args.encoding == 'bert':
	embedding = load_bert_embedding()
	contextualizer = 'average'
else:
	embedding = None
	contextualizer = 'bert'


# instanitae model

if args.model == 'rank':
	model = RankModel([ int(h) for h in args.hidden_dim.split('-')], args.dropout, compostionality_weights, freeze_embeddings=args.freeze_embeddings, bert_layers=args.bert_layers, embedding=embedding, contextualizer=contextualizer)
	criterion = nn.CrossEntropyLoss()
elif args.model == 'sparseBERT':
	criterion = nn.MarginRankingLoss(margin=1)
	model = SparseBERT(args.sparse_dim, args.dropout, args.freeze_bert)


def print_message(s):
	print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


print_message('Starting')

if torch.cuda.device_count() > 1 and not args.single_gpu:
	model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
print(model)
if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")

model = model.to(DEVICE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.0001)


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
		l0_loss = 0
		l1_loss = 0
		loss = 0 
		try:
			features = next(batch_iterator)
		except StopIteration:
			batch_iterator = iter(dataloader_train)
			continue
		if args.model == 'rank':
			out = tuple([model(features['encoded_input'][i].to(DEVICE)) for i in range(dataset_train.num_docs)])
			out = torch.cat(out, 1)
			loss = criterion(out, features['labels'].long().to(DEVICE) * 0)
			train_loss = loss
			total_examples_seen += out.shape[0]
			acc = np.array(((out[:, 0] > out[:,1]).int()).float().cpu().mean())
			# margin ranking loss 
			#train_loss = criterion(out[0], out[1], features['labels'].to(DEVICE))
			#total_examples_seen += out[0].shape[0]
			#acc = np.array(((out[0] > out[1]).int()).float().cpu().mean())
		elif args.model == 'sparseBERT':
			out_queries = model(**features['encoded_queries'].to(DEVICE))
			out_docs_pos = model(**features['encoded_docs'][0].to(DEVICE)) 
			out_docs_neg = model(**features['encoded_docs'][1].to(DEVICE)) 
			out_pos = torch.bmm(out_queries.unsqueeze(1), out_docs_pos.unsqueeze(-1)).squeeze()
			out_neg = torch.bmm(out_queries.unsqueeze(1), out_docs_neg.unsqueeze(-1)).squeeze()
			acc = np.array(((out_pos > out_neg).int()).float().cpu().mean())
			#l1_loss = torch.norm(torch.cat([out_queries, out_docs_pos, out_docs_neg], 0), 1) /3 * args.l1_scalar		
			l1_loss = torch.cat([out_docs_pos, out_docs_neg, out_queries], 1).abs().sum(1).mean() * args.l1_scalar
			l0_loss = (torch.cat([out_docs_pos, out_docs_neg, out_queries], 1) == 0).float().mean(1).mean(0)
			train_loss = criterion(out_pos, out_neg, features['labels'].to(DEVICE))
			loss = train_loss + l1_loss 
			total_examples_seen += out_pos.shape[0]

		total_loss += loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if mb_idx % log_every == 0:
			print(f'MB {mb_idx + 1}/{EPOCH_SIZE}')
			print_message('examples:{}, train_loss:{}, l1_loss: {}, l0_loss: {},  acc:{}'.format(total_examples_seen, train_loss, l1_loss, l0_loss,  acc))
			writer.add_scalar('Loss/Train', train_loss, total_examples_seen)
			writer.add_scalar('Loss/L0', l0_loss, total_examples_seen)
			writer.add_scalar('Loss/Total', loss, total_examples_seen)
			writer.add_scalar('Loss/L1', l1_loss, total_examples_seen)
			writer.add_scalar('Train/Accuracy', acc, total_examples_seen)

		mb_idx += 1
	print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, total_loss / (EPOCH_SIZE) ))
#	model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1  else model.state_dict()
	torch.save(model, MODEL_FILE.format(ep_idx + 1))
	# TESTING
	res_test = {}
	original_scores = {}
	if not args.eval:
		continue
	if ep_idx % test_every == 0:
		model.eval()
		for features in dataloader_test:
			if args.model == 'rank':
				# forward doc
				out = model(features['encoded_input'][0].to(DEVICE))
				# to cpu
			elif args.model == 'sparseBERT':

				out_queries = model(**features['encoded_queries'].to(DEVICE))
				out_docs = model(**features['encoded_docs'][0].to(DEVICE)) 
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



		

