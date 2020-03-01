# nlputils

This is a Python package that implements tools for Natural Language Processing (NLP). These tools are meant to deal with pre-processing for training deep learning models for NLP, such as word segmentation, generating vocabulary, loading dataset, etc.

## Install

```shell
    git clone https://github.com/AlfredWGA/nlputils.git
    cd nlputils
    pip install .
```

## Modules

* `nlputils.tokenizer`  
A module for word segmentation, padding, and converting tokens to integer id.

* `nlputils.vocab_generator`  
Generate a vocabulary given a corpus with tokenized samples.

* `nlputils.data_sequence`  
Inherent from `tf.keras.utils.Sequence`.

* `nlputils.dataset`  
Inherent from `torch.utils.data.Dataset`.

More features would be available in the future.
