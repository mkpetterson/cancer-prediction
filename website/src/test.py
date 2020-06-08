import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

def do_numpy_stuff():
    return np.sqrt(5)

def tokenize(text):
    wordnet = WordNetLemmatizer()
    words = nltk.WhitespaceTokenizer().tokenize(text.lower())
    contractions_removed = [fix_punctuation(word) for word in words]
    stopped_words = [word for word in contractions_removed if word not in stopwords.words('english')]	

    return [wordnet.lemmatize(word) for word in stopped_words]


def fix_punctuation(word):
    word = word.replace('\'', '')
    word = word.replace('.', '')
    return word    


def count_vectorizer(doc):
    vect = CountVectorizer(stop_words='english', tokenizer=nltk.WhitespaceTokenizer().tokenize)
    word_counts = vect.fit_transform(doc)
    word_array = word_counts.toarray()
    return word_array


def tfidf(all_text):
    vect = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    tfid = vect.fit_transform(text)
    tfid_array = tfid.toarray()
    return tfid_array
