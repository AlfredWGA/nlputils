import collections
from .tokenizer import DEFAULT_SPECIAL_TOKENS
from itertools import chain
from collections import Iterable


class VocabGenerator(object):
    """
    A tool class to generate a vocabulary. By default it contains all common special tokens listed in `tokenizer.py`.
    """

    def __init__(self, min_count=None):
        self._min_count = min_count
        self._token2tf = {}
        self._vocab = list(DEFAULT_SPECIAL_TOKENS.values())

    def generate_vocab(self, corpus):
        """
        Generate vocabulary from `corpus`. `corpus` is a list of word lists,
        that contains all the samples from the dataset.

        For example, `corpus` can be [['hello', 'my', 'friend'], ['I', 'feel', 'good']]

        :param corpus: A List of lists.
        :return: The generated vocabulary as a list.
        """
        # Flatten a Iterable object, without spliting string elements.
        flat_samples_iter = chain.from_iterable(item if isinstance(item, Iterable) and
                                                not isinstance(item, str) else [item] for item in corpus)

        # self._count_and_normalize_tf(flat_samples_iter)
        self._token2tf = collections.Counter(flat_samples_iter)
        
        v = self._token2tf.most_common()
        if self._min_count is not None:
            v = filter(lambda x: x[1] > self._min_count, v)
        self._vocab.extend([x[0] for x in v])
        return self._vocab

    def get_vocab(self):
        return self._vocab

    def get_token2tf(self):
        return self._token2tf

    def save_vocab_to(self, fpath):
        """
        Save vocab to fpath.
        """
        with open(fpath, 'w', encoding='utf-8') as f:
            for t in self._vocab:
                f.write(t + '\n')
