from pathlib import Path

__all__ = ['VocabGenerator',
           'BasicTokenizer',
           'pad_sequence_to_fixed_length',
           'EarlyStopping']

DATA_DIR = Path('data/')

SUPPORTED_LANGUAGES = ['cn', 'en']
STOPWORDS_PATH_DICT = {x: DATA_DIR/f'stopwords_{str(x)}.txt' for x in SUPPORTED_LANGUAGES}

from .tokenizer import *
from .vocab_generator import *
from .early_stopping import EarlyStopping
# from .data_sequence import *
# from .dataset import *