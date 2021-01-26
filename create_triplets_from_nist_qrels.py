import argparse
import pickle as p
import numpy as np
from collections import defaultdict

import os



# TREC official relevance scheme:

# Note: Documents were judged on a four-poiunt scale of 
# Not Relevant (0), Relevant (1), Highly Relevant (2) and Perfect (3). 
# Levels 1--3 are considered to be relevant for measures that use binary relevance judgments.
# Passages were judged on a four-point scale of Not Relevant (0), Related (1), Highly Relevant (2), and Perfect (3),
# where 'Related' is actually NOT Relevant---it means that the passage was on the same general topic,
# but did not answer the question. Thus, for Passage Ranking task runs (only), to compute evaluation
# measures that use binary relevance judgments using trec_eval, you either need to use
# trec_eval's -l option [trec_eval -l 2 qrelsfile runfile] or modify the qrels file to change all 1 judgments to 0.


def generate_triplets(args):
	def get_doc_ids(path):
		doc_ids = list()
		with open(path, 'r') as f:
			for line in f:
				doc_id,_ = line.split('\t')
				doc_ids.append(doc_id)
		return doc_ids
				

	out_f_name = args.qrels_file +'_paper_binary'
	if args.docs:
		all_docs = get_doc_ids(args.docs)
		out_f_name += '_rand_docs'

	qrels = defaultdict(lambda: defaultdict(list))

	# arg.pointwise defines whether the generated triplets will be used for 
	# pairiwse training or pointwise training

	# if the qrels will be used for pointwise training
	# then each document-query pair should be used only once

	# if the qrels are going to be used for pairwise training,
	# then we can reuse the same relevant document to create triplets
	# with many irrelevant documets 


	for line in open(args.qrels_file, 'r'):
		line = line.split()

		q_id = line[0]
		d_id = line[2]
		score = int(line[3])

		# if score != 0 and score != 1 and score != 2:
		# 	print(line)


		qrels[q_id][score].append( d_id )

	# qrels do not contain enough irrelevant documents,
	# so we will add some from the top-1000 that are not classified as relevant

	
	out_f = open(out_f_name,  'w')
	

	for q_id in qrels:
		perfectly_relevant = qrels[q_id][3]
		highly_relevant = qrels[q_id][2]
		related = qrels[q_id][1]
		irrelevant = qrels[q_id][0]

		print('perfectly_relevant', len(perfectly_relevant))
		print('highly_relevant', len(highly_relevant))
		print('related', len(related))
		print('irrelevant', len(irrelevant))


		relevant = perfectly_relevant + highly_relevant + related	
		num_of_triplets = len(relevant)
		if args.docs:
			irrelevant = all_docs
		irrelevant = np.random.choice(list(irrelevant), num_of_triplets, replace=False)

		print('relevant', len(relevant))
		print('irrelevant', len(irrelevant)) 
		for rel_id, irrel_id in zip(relevant, irrelevant):
			out_f.write(q_id + '\t' + rel_id + '\t' + irrel_id + "\n")

if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--qrels_file', type=str)
	parser.add_argument('--docs', default=None, type=str)
	args = parser.parse_args()
	    
	for arg in vars(args):
		print(arg, ':', getattr(args, arg))
	generate_triplets(args)
