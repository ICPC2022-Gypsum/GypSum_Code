# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from .batch_handler import vectorize


class CommentDataset(Dataset):
    def __init__(self, examples, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.include_token = config.include_token
        self.examples = examples
        self.args = dict()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.config, self.args)

    def lengths(self):
        if not self.include_token:
            return [(ex.target_len, 0) for ex in self.examples]

        return [(ex.source_len, ex.target_len)
                for ex in self.examples]

    def add_vocab(self, **kwargs):
        self.args = kwargs


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
