"""
Disclaimer: This code is inspired from sklearn implementation.
"""

from collections import defaultdict
from functools import partial
import numpy as np
import scipy as sp


__all__ = [
    "CountVectorizer"
    ]


def _analyze(doc, tokenizer=None, ngram=None):
    
  if tokenizer is not None:
    try:
      doc = tokenizer(doc)
    except Exception as e:
      doc = doc.split(" ")
  
  if ngram is not None:
    doc = ngram(doc)

  return doc


class CountVectorizer:

  def __init__(self, tokenizer=None, binary=False, ngram_range=(1, 1), analyzer="word"):
    self.tokenizer = tokenizer
    self.analyzer = analyzer
    self._vocab = None
    self.binary = binary
    self.ngram_range = ngram_range

  def fit(self, raw_docs, y=None):
    
    """Learn a vocabulary dictionary of all tokens in the raw documents.

    Parameters
    ----------
    raw_docs : iterable
        An iterable which generates str objects.
    y : None
        This parameter is ignored.

    Returns
    -------
    self : object
        Fitted vectorizer.
    """

    self.fit_transform(raw_docs)
    return self
  
  def transform(self, raw_docs):
    
    if not self._vocab:
      raise ValueError("The model has not been fitted yet, therefore no vocabulary available.")
    
    voc, X = self.fit_transform(raw_docs, fixed_vocab=True)

    return voc, X

  def fit_transform(self, raw_docs, fixed_vocab=False):
    """Create sparse feature matrix and vocabulary."""
   
    # Decide whether we have a fixed library or not
    # and create vocab datastructure accordingly
    if fixed_vocab:
      vocabulary = self._vocab
    else:
      # Whenever you encounter new item, assign it with a relevant index
      vocabulary = defaultdict()
      vocabulary.default_factory = vocabulary.__len__

    # Prepare analytical pipeline for docs
    analyze = self._build_analyzer()

    # In order to build the sparse CSR matrix
    # we need to specify according to scipy api
    # below specified parameters
    j_indices = [] # indices of columns
    values = [] # counts of corresponding tons
    indptr = [0] # Helper list keeping track of range of indices and values per doc

    for doc in raw_docs:
        feature_counter = {}
        for feature in analyze(doc):
          try:
            feature_idx = vocabulary[feature]
            if feature_idx not in feature_counter:
                feature_counter[feature_idx] = 1
            else:
                feature_counter[feature_idx] += 1
          except KeyError:
            # This is when we do not want to extend vocab
            continue

        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))

    # disable defaultdict behaviour
    if not fixed_vocab:
      vocabulary = dict(vocabulary)

    # Determine indices dtype
    if indptr[-1] > np.iinfo(np.int32).max:
      indices_dtype = np.int64
    else:
      indices_dtype = np.int32

    # Turn the lists into np arrays
    j_indices = np.asarray(j_indices, dtype=indices_dtype)
    indptr = np.asarray(indptr, dtype=indices_dtype)
    values = np.asarray(values, dtype=np.intc)
    
    # create the sparse matrix
    X = sp.sparse.csr_matrix(
        (values, j_indices, indptr),
        shape=(len(indptr) - 1, len(vocabulary)),
        dtype=np.int64
    )
    
    # Save vocab in case it is new
    if not fixed_vocab:
      self._vocab = vocabulary
    
    # Turn into binary if needed
    if self.binary:
      X.data.fill(1)

    # Sort the indices so you can use it with vocab
    X.sort_indices()

    return vocabulary, X

  def _build_analyzer(self):
    
    if self.analyzer == "word":
      return partial(_analyze, tokenizer=self.tokenizer)
    elif self.analyzer == "char":
      return partial(_analyze, ngram=self._char_ngrams)
    else:
      raise ValueError(f"There is no analyzer called {self.analyzer}. Currently available options are: word and char.")

  
  def _char_ngrams(self, doc):
    """Tokenize doc into a sequence of character n-grams.

    Parameters
    ----------
    doc : str
      Document from which to extract n-gram char features.

    Returns
    -------
    doc : list
      List with tokens.
    """
    
    # Get important input info which will be used later
    text_len = len(doc)
    min_n, max_n = self.ngram_range

    # No need to do any slicing for unigrams, just iterate through the string
    if min_n == 1:
        ngrams = list(doc)
        min_n += 1
    else:
        ngrams = []

    # Bind append method outside of loop to reduce overhead
    ngrams_append = ngrams.append
    
    # Upper boundary practical example:
    # text_len = 20, max_n = 3 --> choose 3
    # text_len = 2, max_n = 3 --> choose 2
    for n in range(min_n, min(max_n + 1, text_len + 1)): 
        for i in range(text_len - n + 1):
            ngrams_append(doc[i : i + n])

    return ngrams

  def get_feature_names(self):
    return np.asarray(
        [t for t, i in sorted(self._vocab.items(), key=lambda x: x[1])],
        dtype=object,
    )

