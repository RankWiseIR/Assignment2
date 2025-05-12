from util import *
import math

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		top_k_docs = query_doc_IDs_ordered[:k]
		# Count how many of the top k documents are in the true_doc_IDs (relevant documents)
		relevant_count = sum(1 for doc_id in top_k_docs if doc_id in true_doc_IDs)
		# Precision is the number of relevant documents in the top k divided by k
		precision = relevant_count / k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		total_precision = 0.0
		num_queries = len(query_ids)
		for i in range(num_queries):
			query_id = query_ids[i]
			predicted_docs = doc_IDs_ordered[i]
			# Extract relevant documents for this query from qrels
			true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			# Compute precision for this query
			precision = self.queryPrecision(predicted_docs, query_id, true_doc_IDs, k)
			total_precision += precision
			
		meanPrecision = total_precision / num_queries if num_queries > 0 else 0.0

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		top_k_docs = query_doc_IDs_ordered[:k]
		# Count how many of the top k documents are in the true relevant documents
		relevant_retrieved = sum(1 for doc_id in top_k_docs if doc_id in true_doc_IDs)
		# Avoid division by zero
		if len(true_doc_IDs) == 0:
			recall = 0.0
		else:
			# Recall is the number of relevant documents retrieved divided by total relevant documents
			recall = relevant_retrieved / len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		total_recall = 0.0
		num_queries = len(query_ids)
		for i in range(num_queries):
			query_id = query_ids[i]
			predicted_docs = doc_IDs_ordered[i]
			# Extract relevant documents for this query from qrels
			true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			# Compute recall for this query
			recall = self.queryRecall(predicted_docs, query_id, true_doc_IDs, k)
			total_recall += recall
		meanRecall = total_recall / num_queries if num_queries > 0 else 0.0


		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		# Calculate precision and recall for the query
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		# Avoid division by zero
		if precision + recall == 0:
			fscore = 0.0
		else:
			# Harmonic mean of precision and recall
			fscore = 2 * (precision * recall) / (precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		total_fscore = 0.0
		num_queries = len(query_ids)
		for i in range(num_queries):
			query_id = query_ids[i]
			predicted_docs = doc_IDs_ordered[i]
			# Extract relevant documents for this query from qrels
			true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			# Compute fscore for this query
			fscore = self.queryFscore(predicted_docs, query_id, true_doc_IDs, k)
			total_fscore += fscore
		meanFscore = total_fscore / num_queries if num_queries > 0 else 0.0

		return meanFscore

	def queryNDCG(self, query_doc_IDs_ordered, query_id, qrels, k):
		# Get graded relevance for this query
		relevance_dict = {}
		for rel in qrels:
			if int(rel["query_num"]) == query_id:
				doc_id = int(rel["id"])
				position = int(rel["position"])
				relevance_dict[doc_id] = position

		# Compute DCG
		DCG = 0.0
		for i, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			rel = relevance_dict.get(doc_id, 0.0)
			# DCG += rel / math.log2(i + 2)
			DCG += (2**rel - 1) / math.log2(i + 2)

		# Compute IDCG (ideal ranking)
		ideal_rels = sorted(relevance_dict.values(), reverse=True)[:k]
		IDCG = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

		return DCG / IDCG if IDCG > 0 else 0.0
			
	
	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		total_nDCG = 0.0
		num_queries = len(query_ids)

		for i in range(num_queries):
			query_id = query_ids[i]
			predicted_docs = doc_IDs_ordered[i]
			nDCG = self.queryNDCG(predicted_docs, query_id, qrels, k)
			total_nDCG += nDCG

		return total_nDCG / num_queries if num_queries > 0 else 0.0

	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, qrels, k):
		"""
		Compute Average Precision@k for a single query using binary relevance from position field.
		"""
		RELEVANCE_THRESHOLD = 1

		# Build set of relevant doc IDs for this query
		relevant_docs = set()
		for rel in qrels:
			if int(rel["query_num"]) == query_id and int(rel["position"]) >= RELEVANCE_THRESHOLD:
				relevant_docs.add(int(rel["id"]))

		num_relevant_found = 0
		sum_precisions = 0.0

		for i, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			if doc_id in relevant_docs:
				num_relevant_found += 1
				precision_at_i = num_relevant_found / (i + 1)
				sum_precisions += precision_at_i

		if len(relevant_docs) == 0:
			return 0.0

		return sum_precisions / min(len(relevant_docs), k)


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Compute MAP@k averaged across all queries.
		"""
		total_AP = 0.0
		num_queries = len(query_ids)

		for i in range(num_queries):
			query_id = query_ids[i]
			predicted_docs = doc_IDs_ordered[i]
			ap = self.queryAveragePrecision(predicted_docs, query_id, qrels, k)
			total_AP += ap

		return total_AP / num_queries if num_queries > 0 else 0.0