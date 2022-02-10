from collections import defaultdict
from functools import partial
import numpy as np
import scipy as sp

__all__ = [
    "CountVectorizer"
    ]


def _analyze(doc, tokenizer=None):
    
  if tokenizer is not None:
    doc = tokenizer(doc)

  return doc


class CountVectorizer:

  def __init__(self, tokenizer, binary=False):
    self.tokenizer = tokenizer
    self.analyzer = "word"
    self._vocab = None
    self.binary = binary

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
      raise ValueError("The model has not been fitted yet, therefore not vocab available.")
    
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

    return vocabulary, X

  def _build_analyzer(self):
    
    if self.analyzer == "word":
      return partial(_analyze, tokenizer=self.tokenizer)
    else:
      raise ValueError(f"There is no analyzer called {self.analyzer} Currently available options are: word.")

  def _analyze(self, doc, tokenizer=None):
    
    if tokenizer is not None:
      doc = tokenizer(doc)

    return doc

