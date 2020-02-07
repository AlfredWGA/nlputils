# coding=utf-8
from tensorflow import keras
import numpy as np


class DataSequence(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=True):
        assert len(x_set) == len(y_set)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.shuffle = shuffle
        self._length = int(np.ceil(len(self.x) / float(self.batch_size)))

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Fetch a batch of samples.
        batch_indices = self.indices[idx * self.batch_size: min(self.__len__(), (idx + 1)) * self.batch_size]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]

        # ....Process.....

        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)