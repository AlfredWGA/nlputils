import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EarlyStopping(object):
    def __init__(self, min_delta=0.0, patience=10, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.stop_flag = False
    
    def _update_counter(self):
        self.counter += 1
        logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.stop_flag = True

    def step(self, score):
        # No initial best score.
        if self.best_score is None:
            self.best_score = score
        
        # Check if score is worse than best score, or is out of patience.
        elif self.mode == 'min':
            if self.best_score <= score - self.min_delta:
                self._update_counter()
        elif  self.mode == 'max':
            if score + self.min_delta <= self.best_score:
                self._update_counter()

        # Achieve a better score.
        else:
            self.best_score = score
            self.counter = 0
        return self.stop_flag