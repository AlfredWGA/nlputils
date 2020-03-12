# coding=utf-8
import jieba
from tqdm import tqdm
import spacy
import pickle
import numpy as np
import collections
from collections.abc import Iterable
import logging
from pathlib import Path
from . import STOPWORDS_PATH_DICT
logger = logging.getLogger(__name__)


# Default values for common special tokens
DEFAULT_SPECIAL_TOKENS = {
    "bos_token": '[BOS]',
    "eos_token": '[EOS]',
    "unk_token": '[UNK]',
    "sep_token": '[SEP]',
    "pad_token": '[PAD]',
    "cls_token": '[CLS]',
    "mask_token": '[MASK]'
}


def pad_sequence_to_fixed_length(sequence, max_length, value=0, padding_mode='right', truncate_mode='right'):
    """
    Pad a sequence (Iterable) using `value` to fixed length. List longer than `max_length` would
    be truncated. `padding_mode` and `truncate_mode` can be either `right` or `left`.

    :param sequence: A Iterable object.
    :param max_length: Max length to pad.
    :param value: Value for padding. 
    :param padding_mode: Specify which side to pad. Either `right` or `left`.
    :param truncate_mode: Which side to truncate.
    :return: A list.
    """
    assert isinstance(sequence, Iterable) and not isinstance(sequence, str)
    supported_mode = ['right', 'left']
    assert padding_mode in supported_mode and truncate_mode in supported_mode

    sequence = list(sequence)
    if len(sequence) >= max_length:
        if truncate_mode == 'right':
            return sequence[:max_length]
        elif truncate_mode == 'left':
            return sequence[len(sequence)-max_length:]

    else:
        paddings = [value for _ in range(
            max_length - len(sequence))]
        if padding_mode == 'right':
            return sequence + paddings
        elif padding_mode == 'left':
            return paddings + sequence


class Tokenizer(object):
    """
    A abstract class for tokenizer. Behaviors such as tokenization and token-id mapping can be customized to fit
    a variety of task and models.

    To write your own custom tokenizer, the following methods must be override: 

    ```
    _convert_token_to_id(token)

    _convert_id_to_token(index)

    tokenize(string)

    convert_tokens_to_ids(tokens)

    convert_ids_to_tokens(ids)
    ```

    """

    def __init__(self, **kwargs):
        """
        Common and additional special tokens attributes can be specified using keyword arguments.

        self._SPECIAL_TOKENS_ATTRIBUTES = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens"]

        :param kwargs:
        """
        self._SPECIAL_TOKENS_ATTRIBUTES = [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
            "additional_special_tokens"
        ]

        self._bos_token = DEFAULT_SPECIAL_TOKENS["bos_token"]
        self._eos_token = DEFAULT_SPECIAL_TOKENS["eos_token"]
        self._unk_token = DEFAULT_SPECIAL_TOKENS["unk_token"]
        self._sep_token = DEFAULT_SPECIAL_TOKENS["sep_token"]
        self._pad_token = DEFAULT_SPECIAL_TOKENS["pad_token"]
        self._cls_token = DEFAULT_SPECIAL_TOKENS["cls_token"]
        self._mask_token = DEFAULT_SPECIAL_TOKENS["mask_token"]
        self._additional_special_tokens = []

        self._vocab = []
        self._ids = []
        self._token2id = {}
        self._id2token = {}

        for key, value in kwargs.items():
            if key in self._SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens' and not isinstance(value, Iterable):
                    raise ValueError(
                        'Value of additional_special_tokens must be Iterable.')
                elif key != 'additional_special_tokens' and not isinstance(value, str):
                    raise ValueError(
                        'Value of any special tokens must be string.')
                setattr(self, '_' + key, value)
            else:
                raise ValueError(
                    'Not supported keyword argument {}.'.format(key))

    def _convert_token_to_id(self, token):
        """
        Convert a token to an id.
        """
        raise NotImplementedError

    def _convert_id_to_token(self, index):
        """
        Convert an id to a token.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """
        Convert a list of tokens to a list of ids.
        """
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids):
        """
        Convert a list of id to a list of tokens.
        """
        raise NotImplementedError

    def tokenize(self, string):
        """
        Tokenize a string.
        """
        raise NotImplementedError

    def save_to(self, fpath):
        """
        Save the tokenizer object as pickle file.
        """
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    def load_vocab(self, src):
        """
        Load vocab from either a Iterable object or a file path.
        """
        raise NotImplementedError

    def encode(self, string, max_length=None, padding_mode='right', truncate_mode='right'):
        """
        Encode a string into a list of ids.

        Same as doing `tokenizer.convert_tokens_to_ids(pad_sequence_to_fixed_length(
            self.tokenize(string), ...)`

        :param string: A string to be converted.
        :param max_length: Max length for a token list. `None` if no padding or truncation.
        :param padding_mode: Specify which side to pad. Either `right` or `left`.
        :param truncate_mode: Which side to truncate.
        :return: A list of ids.
        """
        tokens = self.tokenize(string)
        ids = self.convert_tokens_to_ids(tokens)
        if max_length is not None:
            ids = pad_sequence_to_fixed_length(
                ids, max_length, self._convert_token_to_id(self._pad_token),
                padding_mode, truncate_mode)
        return ids

    def decode(self, ids):
        """
        Decode a list of ids into a string.

        :param ids: List of id.
        :return: The decoded string.
        """
        return ''.join(self.convert_ids_to_tokens(ids))

    def get_vocab(self):
        """
        Return the vocabulary as a set.
        """
        return self._vocab

    def get_token2id(self):
        """
        Return a `dict`.  
        """
        return self._token2id

    def get_id2token(self):
        """
        Return a `dict`.  
        """
        return self._id2token

    def get_all_special_tokens(self):
        return [getattr(self, '_' + x) for x in self._SPECIAL_TOKENS_ATTRIBUTES]


class BasicTokenizer(Tokenizer):
    """
    A basic tokenizer. Support word segmentation and loading stop words.
    By default, Chinese stopwords are provided. English, stopwords are loaded from `nltk`.


    Supported languages: `cn`, `en`   
    """

    def __init__(self, language='cn', norm=False, merge_ne=False, **kwargs):
        """
        `lemma` and 'merge_ne' only valid for `en`.
        """
        self._SUPPORTED_LANGUAGE = {'cn', 'en'}
        if language not in self._SUPPORTED_LANGUAGE:
            raise ValueError(f'Language {language} not supported.')

        if language == 'en':
            self.norm = norm
            self.model = spacy.load('en_core_web_sm')
            if merge_ne:
                # Merge name entities.
                merge_ents = self.model.create_pipe("merge_entities")
                self.model.add_pipe(merge_ents)

        self._stop_words = self._init_stop_words(language)
        self._language = language
        super(BasicTokenizer, self).__init__(**kwargs)
    
    def _init_stop_words(self, language):
        fpath = Path(__file__).parent / STOPWORDS_PATH_DICT[language]
        with open(fpath, 'r', encoding='utf-8') as f:
            stop_words = set([line.strip() for line in f])  
        logger.info('Load stopwords from {}'.format(fpath))    
        return stop_words

    def _convert_token_to_id(self, token):
        return self._token2id[token]

    def _convert_id_to_token(self, index):
        return self._id2token[index]

    def convert_tokens_to_ids(self, tokens):
        """
        Convert a list of tokens to a list of ids. Words that are not in the vocabulary
        would be replaced by `self._unk_token`. Return a list with the same length.
        """
        tokens_ = []
        for t in tokens:
            if t not in self._vocab:
                tokens_.append(self._unk_token)
            else:
                tokens_.append(t)

        return [self._convert_token_to_id(t) for t in tokens_]

    def convert_ids_to_tokens(self, ids):
        ids_ = []
        for i in ids:
            if i not in self._ids:
                ids_.append(self._token2id[self._unk_token])
            else:
                ids_.append(i)
        return [self._convert_id_to_token(i) for i in ids]

    def discard_stop_words(self, tokens):
        return [t for t in tokens if t not in self._stop_words]

    def tokenize(self, string, no_stop_words=False):
        """
        Tokenize a string.

        :param string: A string.
        :param no_stop_words: Set `True` to remove stop words from the string.
        :return:
        """
        if self._language == 'cn':
            tokens = list(jieba.cut(string))

        elif self._language == 'en':
            if self.norm is True:
                tokens = [t.norm_ for t in self.model(string)]
            else:
                tokens = [t.text for t in self.model(string)]

        if no_stop_words:
            tokens = self.discard_stop_words(tokens)
        return tokens

    def encode(self, string, max_length=None, padding_mode='right', truncate_mode='right', no_stop_words=False):
        """
        Encode a string into a list of ids.

        Same as doing `tokenizer.convert_tokens_to_ids(pad_sequence_to_fixed_length(
            self.tokenize(string), ...)`

        :param string: A string to be converted.
        :param max_length: Max length for a token list. `None` if no padding or truncation.
        :param padding_mode: Specify which side to pad. Either `right` or `left`.
        :param truncate_mode: Which side to truncate.
        :return: A list of ids.
        """
        tokens = self.tokenize(string, no_stop_words)
        ids = self.convert_tokens_to_ids(tokens)
        if max_length is not None:
            ids = pad_sequence_to_fixed_length(
                ids, max_length, self._convert_token_to_id(self._pad_token),
                padding_mode, truncate_mode)
        return ids

    def load_vocab(self, src):
        """
        Load vocab from either a Iterable object or a file path. Ids (integer) are
        generated using `range(len(self._vocab))`.
        """
        if not isinstance(src, (Iterable, str)):
            raise ValueError(
                'Vocab can only be loaded from Iterable or file path.')

        if isinstance(src, str):
            with open(src, 'r', encoding='utf-8') as f:
                self._vocab = [x.strip() for x in f.readlines()]
            logger.info('Load vocabulary from {}.'.format(src))

        else:
            self._vocab = list(src)

        # Assign id (integer) for each token in the vocabulary.
        self._ids = [i for i in range(len(self._vocab))]
        self._token2id = {self._vocab[i]: i for i in self._ids}
        self._id2token = {value: key for key, value in self._token2id.items()}

        # Cast vocab and ids to set().
        # self._vocab = set(self._vocab)
        # self._ids = set(self._ids)
        return self.get_vocab()

    def load_stopwords(self, stopwords_path):
        """
        Load custom stopwords into the tokenizer. Only UTF-8 encoding files.
        :param stop_words_path: File to load from, with one stopword each line.
        """
        with open(stopwords_path, mode='r', encoding='utf-8') as f:
            self._stop_words = set([x.strip() for x in f])
        logger.info('Load stopwords from {}.'.format(stopwords_path))

    def get_stopwords(self):
        """
        Return a list of stopwords.
        """
        return list(self._stop_words)
