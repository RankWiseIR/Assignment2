from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

from collections import Counter 

class StopwordRemoval():

	def fromList(self, text):
		"""
		Stop word removal

		Parameters
		----------
		text : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here
		stop_words = set(stopwords.words('english'))
		stopwordRemovedText = [[token for token in sent if token.lower() not in stop_words] for sent in text]

		return stopwordRemovedText



	