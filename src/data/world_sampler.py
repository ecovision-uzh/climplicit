import torch

from tqdm import tqdm

from typing import Iterator, Sequence

import numpy as np

"""class WeightedRandomWorldSampler(torch.utils.data.sampler.Sampler[int]):
    #Adapted from the original WeightedRandomSampler
    def __init__(self, weights: Sequence[float], num_samples: int) -> None:
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        self.weights = weights_tensor
        self.num_samples = num_samples
        
        self.BLOCK_SIZE = 256 # 32 #1024  #32
        num_blocks = (self.num_samples // self.BLOCK_SIZE) + 1
        self.block_weights = torch.tensor([self.weights[i*self.BLOCK_SIZE:(i+1)*self.BLOCK_SIZE].sum() for i in range(num_blocks)], dtype=torch.double)
        
        self.SAMPLING_LEN = 1024

    def __iter__(self) -> Iterator[int]:
        #block_ids_sample = torch.multinomial(self.block_weights, self.num_samples, replacement=True)
        block_ids_sample = torch.multinomial(self.block_weights, self.SAMPLING_LEN, replacement=True)
        s_counter = 0
        for _ in range(self.num_samples):
            if s_counter == self.SAMPLING_LEN:
                s_counter = 0
                block_ids_sample = torch.multinomial(self.block_weights, self.SAMPLING_LEN, replacement=True)
            block_id = block_ids_sample[s_counter]
            s_counter += 1
            t, f = block_id*self.BLOCK_SIZE, (block_id+1)*self.BLOCK_SIZE
            yield torch.multinomial(self.weights[t:f], 1, replacement=True)
        #yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples"""


class WeightedRandomWorldSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_samples):
        self.num_samples = num_samples

        self.num_blocks = (len(weights) // pow(2, 24)) + 1
        self.block_probabilities = torch.tensor(
            [
                sum(weights[pow(2, 24) * i : pow(2, 24) * (i + 1)]) / len(weights)
                for i in range(self.num_blocks)
            ]
        )
        # block_probabilities = [p/sum(block_probabilities) for p in block_probabilities]
        self.weights = weights
        self.block_samplers = {
            i: torch.utils.data.WeightedRandomSampler(
                weights=weights[pow(2, 24) * i : pow(2, 24) * (i + 1)],
                num_samples=int(self.num_samples * self.block_probabilities[i]),
                replacement=True,
            )
            for i in range(self.num_blocks)
        }
        self.block_samplers_iter = {
            i: iter(self.block_samplers[i]) for i in range(self.num_blocks)
        }
        # self.block_samplers = {i: torch.multinomial(torch.tensor(weights[pow(2, 24)*i:pow(2, 24)*(i+1)]),
        #                        int(self.num_samples * self.block_probabilities[i] * oversample))
        #                        for i in range(self.num_blocks)}

        self.block_id_sampler = iter(
            torch.utils.data.WeightedRandomSampler(
                weights=self.block_probabilities,
                num_samples=self.num_samples,
                replacement=True,
            )
        )
        # self.block_id_sampler = iter(torch.multinomial(self.block_probabilities, self.num_samples, replacement=True))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        sampled = {i: 0 for i in range(self.num_blocks)}
        for block_id in self.block_id_sampler:
            bid = int(block_id)
            if sampled[bid] == int(self.num_samples * self.block_probabilities[bid]):
                self.block_samplers_iter[bid] = iter(self.block_samplers[bid])
                sampled[bid] = 0
            yield next(self.block_samplers_iter[bid])
            sampled[bid] += 1


if __name__ == "__main__":
    weights = np.load("/shares/wegner.ics.uzh/CHELSA/input/idx_to_weight.npy")
    num_samples = len(weights)

    from time import time

    start = time()

    sampler = iter(WeightedRandomWorldSampler(weights, num_samples))

    del weights

    for i in tqdm(range(num_samples)):
        next(sampler)

    print("Took:", time() - start)
