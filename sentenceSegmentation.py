from util import *

# Add your import statements here
import re
import nltk
from nltk.tokenize import PunktTokenizer
# nltk.download('all') # ensures that all pretrained models are downloaded

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText = re.split(r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bJr)(?<!\bSr)(?<!\bInc)(?<!\bLtd)(?<!\bvs)\.|\?|!', text)
		segmentedText = [s.strip() for s in segmentedText if s.strip()]
		
		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		tokenizer = PunktTokenizer()
		segmentedText = tokenizer.tokenize(text.strip())
		
		return segmentedText