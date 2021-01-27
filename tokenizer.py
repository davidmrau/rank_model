import numpy as np
import argparse
import pickle
from transformers import BertTokenizerFast
import nltk
from nltk.stem import PorterStemmer

import os
import sys

import unicodedata
from unidecode import unidecode

class Tokenizer():

	def __init__(self, tokenizer="bert", max_len=-1, stopwords="lucene", remove_unk = True, dicts_path = "data/embeddings/",
					lower_case=True, unk_words_filename = None, stemmer = False):
		"""
		Stopwords:
			"none": Not removing any stopwords
			"lucene": Remove the default Lucene stopwords
			"some/path/file": each stopword is in one line, in lower case in that txt file
		"""


		if tokenizer != "bert" and tokenizer != "glove" and tokenizer != 'msmarco' and tokenizer != 'robust' and tokenizer != 'word2vec' :
			raise ValueError("'tokenizer' param not among {bert/glove/msmarco/robust} !")


		self.lower = lower_case
		self.tokenizer = tokenizer
		if stemmer:
			self.stemmer = PorterStemmer()
		else:
			self.stemmer = None

		if self.tokenizer == "bert":
			self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		elif self.tokenizer == "msmarco":
			self.word2idx = pickle.load(open(dicts_path + 'msmarco_ascii.tsv.vocab_count_400000_t2i.p', 'rb'))
			self.idx2word = pickle.load(open(dicts_path + 'msmarco_ascii.tsv.vocab_count_400000_i2t.p', 'rb'))
		elif self.tokenizer == "word2vec":
			self.word2idx = pickle.load(open(dicts_path + 'GoogleNews-vectors-negative300.word2idx.p', 'rb'))
			self.idx2word = pickle.load(open(dicts_path + 'GoogleNews-vectors-negative300.idx2word.p', 'rb'))
		elif self.tokenizer == "glove":
			self.word2idx = pickle.load(open(dicts_path + 'glove.6B.300d_word2idx_dict.p', 'rb'))
			self.idx2word = pickle.load(open(dicts_path + 'glove.6B.300d_idx2word_dict.p', 'rb'))
		elif self.tokenizer == 'robust':
			self.word2idx = pickle.load(open(dicts_path + 'robust_vocab.p', 'rb'))
		self.max_len = max_len
		self.remove_unk = remove_unk
		self.unk_words_filename = unk_words_filename

		# delete older version of the file if there exists one
		if self.unk_words_filename is not None and os.path.isfile(self.unk_words_filename):
			print("Older version of unk words file was found. It will be deleted and updated.")
			os.remove(self.unk_words_filename) 


		self.unk_word_id = None
		self.set_unk_word(remove_unk = remove_unk)
		self.set_stopword_ids_list(stopwords = stopwords)


	def stanford_tokenize(self, text):
		document = self.client.annotate(text)
		tokenized = list()
		for sent in document.sentence:
			tokenized += [token.word.lower() for token in sent.token]
		return tokenized
		
	def set_unk_word(self, remove_unk):

		if self.tokenizer == "bert":
			self.unk_word = "[UNK]"
		elif self.tokenizer == "word2vec":
			self.unk_word = "unk"
		elif self.tokenizer == "glove":
			self.unk_word = "unk"

		elif self.tokenizer == "robust":
			self.unk_word = "<unk>"
		elif self.tokenizer == "msmarco":
			self.unk_word = "<unk>"

		self.unk_word_id = self.get_word_ids(self.unk_word)

		if remove_unk:
			self.stopword_ids_list = list(self.unk_word_id)


	def set_stopword_ids_list(self, stopwords):
		if stopwords == "none":
			self.stopword_ids_list = []
			self.stopwords_list = []

		elif stopwords == "lucene":
			# If not specified, using standard Lucene stopwords list

			# Lucene / anserini default stopwords:
			# https://stackoverflow.com/questions/17527741/what-is-the-default-list-of-stopwords-used-in-lucenes-stopfilter
			# The default stop words set in StandardAnalyzer and EnglishAnalyzer is from StopAnalyzer.ENGLISH_STOP_WORDS_SET:
			# https://github.com/apache/lucene-solr/blob/master/lucene/analysis/common/src/java/org/apache/lucene/analysis/en/EnglishAnalyzer.java#L46

			lucene_stopwords_list = ["a", "an", "and", "are", "as", "at", "be", "but", "by",
			"for", "if", "in", "into", "is", "it",
			"no", "not", "of", "on", "or", "such",
			"that", "the", "their", "then", "there", "these",
			"they", "this", "to", "was", "will", "with"]


			self.stopwords_list = lucene_stopwords_list


			self.stopword_ids_list = [self.get_word_ids(word.lower()) for word in self.stopwords_list]


		else:
			raise ValueError("Implement function to read stopwords from provided 'stopwords' argument!")


	def get_word_ids(self, word):
		
		if self.tokenizer == "bert":
			token_id = self.bert_tokenizer.encode(word)[1:-1]
			if self.unk_word_id:
				if token_id[0] == self.unk_word_id:
					with open(self.unk_words_filename, "a") as myfile:
						myfile.write(word + "\n")
			return token_id

		else:
			if word in self.word2idx:
				return [self.word2idx[word]]
			else:
				# if selected, the unknown words that are found, are being written to the specified file, line by line
				if self.unk_words_filename is not None:
					with open(self.unk_words_filename, "a") as myfile:
						myfile.write(word + "\n")
				return [self.word2idx[self.unk_word]]

	def encode(self, text):
		""" Remove stopwords, tokenize and translate to word ids for a given text
		"""
		if self.lower:
			text = text.lower()

		tokens = nltk.word_tokenize(text)

		if self.stemmer is not None:
			tokens = [self.stemmer.stem(token) for token in tokens]

		#tokens = self.stanford_tokenize(text[:100000])
		# if there is a specified max len of input to be considered, we enforce it on the token level
		# as the token level is percieved by nltk.word_tokenize()
		if self.max_len != -1:
			tokens = tokens[:self.max_len]

		token_ids = []

		for i, token in enumerate(tokens):
			# if the token is not among the stopwords
			if token not in self.stopwords_list:
				# we added on the resulting token ids list
				token_ids += self.get_word_ids(token)

		return token_ids



	def decode(self, word_ids):

		if self.tokenizer == "bert":
			return self.bert_tokenizer.decode(word_ids)

		else:
			# translate into words in a string split by ' ' and return it
			return ' '.join(self.idx2word[word_id] for word_id in word_ids)



	def get_word_from_id(self,word_id):
		if self.tokenizer == "bert":
			return self.bert_tokenizer.decode(word_id)
		else:
			return self.idx2word[word_id]

# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def tokenize(args):

	in_fname = args.input_file

	print("Tokenizing :", in_fname)
	# add = 'glove' if args.whitespace else 'bert'
	out_fname = f'{in_fname}_{args.tokenizer}_stop_{args.stopwords}{"_remove_unk" if args.remove_unk else ""}' + \
				f'{"_max_len_" + str(args.max_len) if args.max_len != -1 else "" }{"_stemmed" if args.stemmer else ""}.tsv'
	print("To file    :", out_fname)

	if args.dont_log_unk:
		unk_words_filename = None
	else:
		unk_words_filename = out_fname + "_unk_words"

	tokenizer = Tokenizer(tokenizer = args.tokenizer, max_len = args.max_len, stopwords=args.stopwords,
							remove_unk = args.remove_unk, dicts_path = args.dicts_path, unk_words_filename = unk_words_filename,
							stemmer = args.stemmer)

	empty_ids_filename = out_fname + "_empty_ids"

	# word2idx = pickle.load(open(args.word2index_path, 'rb'))
	with open(out_fname, 'w') as out_f:
		with open(in_fname, 'r', encoding=args.encoding) as in_f:
			with open(empty_ids_filename, 'w') as empty_ids_f:
				for count, line in enumerate(in_f):
					if count % 1000 == 0 and count != 0:
						print(f'lines read: {count}')

					spl = line.strip().split(args.delimiter, 1)
					if len(spl) < 2:
						id_ = spl[0].strip()

						# writing ids of text that is empty before tokenization
						empty_ids_f.write(id_ + "\t\n")

						out_f.write(id_ + '\t\n')
						continue
					
					id_, text = spl
					id_ = id_.strip()
					text = text.strip()
					text = unidecode(text)
					tokenized_ids = tokenizer.encode(text)
					print(text)
					print(tokenized_ids)
					if len(tokenized_ids) == 0:
						# writing ids of text that is empty after tokenization
						empty_ids_f.write(id_ + "\t\n")
						out_f.write(id_ + '\t\n')
						continue

					out_f.write(id_ + '\t' + ' '.join(str(t) for t in tokenized_ids) + '\n')




def detokenize(args):

	in_fname = args.input_file

	print("Decoding :", in_fname)
	# add = 'glove' if args.whitespace else 'bert'

	tokenizer = Tokenizer(tokenizer = args.tokenizer)


	# word2idx = pickle.load(open(args.word2index_path, 'rb'))
	with open(in_fname, 'r', encoding=args.encoding) as in_f:
			for count, line in enumerate(in_f):

				spl = line.strip().split(args.delimiter, 1)
				id_, text = spl
				id_ = id_.strip()
				text = text.strip()
				tokens = np.fromstring(text, dtype=int, sep=' ')
				
				tokenized_ids = tokenizer.decode(tokens)

				print(id_ + '\t' + tokenized_ids.encode('latin-1').decode('utf-8'))
if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--delimiter', type=str, default='\t')
	parser.add_argument('--input_file', type=str)
	parser.add_argument('--encoding', default=None, type=str)
	parser.add_argument('--max_len', default=-1, type=int)
	parser.add_argument('--dicts_path', type=str, default='data/embeddings/')
	parser.add_argument('--tokenizer', type=str, help = "{'bert','glove', 'robust', 'msmarco', 'word2vec'}")
	parser.add_argument('--stopwords', type=str, default="none", help = "{'none','lucene', 'some/path/file'}")
	parser.add_argument('--remove_unk', action='store_true')
	parser.add_argument('--stemmer', action='store_true')
	parser.add_argument('--decode', action='store_true')
	parser.add_argument('--dont_log_unk', action='store_true')
	args = parser.parse_args()

	if not args.decode:	
		tokenize(args)
	else:
		detokenize(args)
