import copy
import random
from collections import defaultdict

from torch.utils.data import Sampler


class C2IDistributedSampler(Sampler):
    def __init__(
            self,
            labels: list[int],
            num_replicas: int,
            rank: int,
            seed: int = 0,
    ):
        super().__init__()
        self.labels = labels
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.indices[label].append(idx)
        self.classes = list(sorted(self.indices.keys()))

        self.num_samples = dict()
        self.total_size = dict()
        for c in self.classes:
            num_samples = len(self.indices[c]) // self.num_replicas
            total_size = num_samples * self.num_replicas
            self.num_samples[c] = num_samples
            self.total_size[c] = total_size

    def get_indices(self):
        indices = copy.deepcopy(self.indices)
        rng = random.Random(self.seed + self.epoch)
        for c in self.classes:
            # shuffle
            rng.shuffle(indices[c])
            # drop last
            indices[c] = indices[c][:self.total_size[c]]
            assert len(indices[c]) == self.total_size[c]
            # subsample
            indices[c] = indices[c][self.rank : self.total_size[c] : self.num_replicas]
            assert len(indices[c]) == self.num_samples[c]
        return indices

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class C2IBatchSampler(Sampler):
    def __init__(
            self,
            sampler: C2IDistributedSampler,
            num_classes_per_batch: int,
            num_samples_per_class: int,
    ):
        super().__init__()
        self.sampler = sampler
        self.classes = sampler.classes
        self.num_classes_per_batch = num_classes_per_batch
        self.num_samples_per_class = num_samples_per_class

    def __iter__(self):
        rng = random.Random(self.sampler.seed + self.sampler.epoch)
        indices = self.sampler.get_indices()
        remain_count = {c: len(indices[c]) for c in self.classes}
        available_classes = {c for c in self.classes}
        while len(available_classes) >= self.num_classes_per_batch:
            batch_classes = rng.sample(sorted(available_classes), self.num_classes_per_batch)
            batch_indices = []
            for c in batch_classes:
                batch_indices.extend(indices[c][:self.num_samples_per_class])
                indices[c] = indices[c][self.num_samples_per_class:]
                remain_count[c] -= self.num_samples_per_class
                if remain_count[c] < self.num_samples_per_class:
                    available_classes.remove(c)
            yield batch_indices
