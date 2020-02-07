import collections
from .tokenizer import DEFAULT_SPECIAL_TOKENS
from itertools import chain
from collections import Iterable


class VocabGenerator(object):
    """
    A tool class to generate a vocabulary. By default it contains all common special tokens listed in this file.
    Pass the `coverage` parameter to indicate how much of the whole corpus the vocabulary should cover.

    First a initial vocabulary are generated from the corpus, with all words sorted by term frequencies. Then we
    keep the top-N most frequent words that can cover at least X% of the total term frequency. X is specified by 
    `coverage`.      

    Through this you can omit rare words with low frequencies in the vocabulary.
    For example, 1.0 indicates the whole vocabulary (without omission). 0.85 indicates that
    we only keep words that takes up 85% of the total term frequencies.

    """
    def __init__(self, coverage=1.0):
        self._coverage = coverage
        self._token2tf = {}
        self._normalized_token2tf = {}
        self._vocab = [
            DEFAULT_SPECIAL_TOKENS["pad_token"],
            DEFAULT_SPECIAL_TOKENS["bos_token"],
            DEFAULT_SPECIAL_TOKENS["eos_token"],
            DEFAULT_SPECIAL_TOKENS["unk_token"],
            DEFAULT_SPECIAL_TOKENS["cls_token"],
            DEFAULT_SPECIAL_TOKENS["sep_token"],
            DEFAULT_SPECIAL_TOKENS["mask_token"]
        ]

    def _count_and_normalize_tf(self, corpus):
        """
        Count term frequency for each word, and normalize them to between 0~1.

        """
        self._token2tf = collections.Counter(corpus)
        total_tf = sum(self._token2tf.values())
        self._normalized_token2tf = dict(
            [(token, tf / total_tf) for token, tf in self._token2tf.items()])

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

        self._count_and_normalize_tf(flat_samples_iter)
        sorted_tokens_with_tf = sorted(self._normalized_token2tf.items(),
                        key=lambda x: x[1], reverse=True)

        stop_index = 0
        coverage = 0.0
        for _, tf in sorted_tokens_with_tf:
            coverage += tf
            stop_index += 1
            if coverage >= self._coverage:
                break

        self._vocab.extend([x[0] for x in sorted_tokens_with_tf[:stop_index]])
        return self._vocab

    def get_vocab(self):
        return self._vocab
    def save_vocab_to(self, fpath):
        """
        Save vocab to fpath.
        """
        with open(fpath, 'w', encoding='utf-8') as f:
            for t in self._vocab:
                f.write(t + '\n')