from util import *

# Add your import statements here
import numpy as np

EPSILON = 1e-10

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		index = {}
		for doc_id, doc in zip(docIDs, docs):
			term_freq = {}
			for sent in doc:
				for term in sent:
					term_freq[term] = term_freq.get(term, 0) + 1
			index[doc_id] = term_freq

		self.index = index
	

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		# TODO - punctuation removal

		#Fill in code here

		# create list of unique tokens in corpus
		tokens = list(set([term for term_dict in self.index.values() for term in term_dict.keys()]))
		token2idx = {token: idx for idx, token in enumerate(tokens)}
		idx2token = {value: key for key, value in token2idx.items()}

		# create term frequency matrix
		tf = np.zeros((len(tokens), len(self.index)))
		# populate term frequency matrix
		for doc_id, term_freq in self.index.items():
			for term, freq in term_freq.items():
				tf[token2idx[term], doc_id-1] = freq	# index with doc_id - 1 to get 0-based index
		
		# sanity check
		# print(len(tokens) * len(self.index), (tf == 0).sum(), (tf > 0).sum())

		# create inverse document frequency vector
		idf = np.zeros(len(tokens))
		for term in tokens:
			idf[token2idx[term]] = np.log(len(self.index) / sum([1 for terms in self.index.values() if term in terms.keys()]))

		# sanity check
		# for i in range(len(tokens)):
		# 	print(idf[i])

		# create tf-idf matrix
		tf_idf =tf * idf[:, np.newaxis]

		# sanity check
		# print(np.isnan(tf_idf).sum(), (tf_idf == 0).sum())

		# create tf matrix for queries
		q_tf = np.zeros((len(tokens), len(queries)))
		for idx, query in enumerate(queries):
			for sent in query:
				for term in sent:
					if term in token2idx.keys():
						q_tf[token2idx[term], idx] += 1
						
		# sanity check
		# print(len(tokens) * len(queries), (q_tf == 0).sum(), (q_tf > 0).sum())

		# create tf-idf matrix for queries
		q_tf_idf = q_tf * idf[:, np.newaxis]

		# sanity check
		# print(np.isnan(q_tf_idf).sum(), (q_tf_idf == 0).sum(), (q_tf_idf > 0).sum())

		# calculate cosine similarity between each query and each document
		similarity = np.zeros((len(queries), len(self.index)))
		for i in range(len(queries)):
			for j in range(len(self.index)):
				denominator = np.linalg.norm(q_tf_idf[:, i]) * np.linalg.norm(tf_idf[:, j])
				if denominator == 0:
					# if denominator is 0, set similarity to 0
					similarity[i, j] = 0
				else:
					similarity[i, j] = np.dot(q_tf_idf[:, i], tf_idf[:, j]) / denominator

			# rank the documents for each query
			ranked_docIDs = np.argsort(similarity[i])[::-1]
			# convert to 1-based index
			ranked_docIDs = [doc_id + 1 for doc_id in ranked_docIDs]
			doc_IDs_ordered.append(ranked_docIDs)

		# sanity check
		# print(similarity.shape[0] * similarity.shape[1], (similarity == 0).sum(), (similarity > 0).sum(), similarity.max(), similarity.min(), np.isnan(similarity).sum())	
		
		return doc_IDs_ordered




