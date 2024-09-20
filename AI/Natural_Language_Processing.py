''' It's a field of AI that deals with how computers and humans interact and how to program computers to process and
analyze huge amounts of natural language data.
There are two main Componets of NLP
1. Natural Language Understanding (NLU): Resolves around machine reading comprehension
2. Natural Languge Generation(NLG): It's the process of producing meaningful phrases and sentences in the form of natural
language from a representation system like a knowledge base or a logical form'''
import nltk
# Download the required data
#nltk.download()
#nltk.download('punkt')
#nltk.download('wordnet')
# Tokennization basically means breaking down data into smaller chucks or tokens, so that they can be easily analyzed
# word_tokenize package divides the input text into words
from nltk.tokenize import word_tokenize
text = "One day or day one! It's your choice."
print(word_tokenize(text))
# sent_tokenize package:  this package will divide the input into sentences
from nltk.tokenize import sent_tokenize
print(sent_tokenize(text))
# Stemming is the process of cutting off the prefixes and suffixes of the word and taking into account only the root word
# For example: the words, waited, waiting and waits can all be trimmed down to their root word, i.e. wait
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# Test the stemmer on various pluralised words
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned', 'humbled', 'sized', 'meeting',
           'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference', 'colonizer', 'plotted']
singles = [stemmer.stem(plural) for plural in plurals]
print(' '.join(singles))
# Lemmatization is similar to stemming however, it's more effective because it takes into consideration the morphological
# analysis of the words
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))
# A denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos = "a"))
# Bag of Word (BoW) is used to extract the features from text so that the text can be used in modeling such that in
# machine learning algorithms
