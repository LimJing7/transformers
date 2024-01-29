# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Copyright 2023 DSO National Laboratories.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Arithmetic Sampling
# https://proceedings.mlr.press/v202/vilnis23a/vilnis23a.pdf
# Code adapted from https://github.com/google-research/google-research/tree/master/arithmetic_sampling
# ------------------------------------------------------------------------------


def arithmetic_categorical(logits, codes):
    """Sample from a categorical using arithmetic sampling.

    Returns samples from an arithmetic codebook based on provided codes. This gives an unbiased sample for each code
    randomly picked from the unit interval.

    Args:
        logits: array: [batch_size, vocab_size] float32 sequence of logits.
        codes: array: [batch_size] float32 codes for each batch element.

    Returns:
        A tuple (samples, new_codes) where `samples` are sampled indices with shape [batch_size], and `new_codes` are
        shape [batch_size] containing codes for the remaining suffix if doing ancestral sampling.
    """
    # We randomly permute the logits here at each timestep to avoid depending on
    # The default order of the vocabulary. This isn't strictly necessary.
    # We need to invert this permutation at the end cause it changes the
    # identities of the sampled indices.
    # Permutation has been turned off for efficiency reasons

    _, vocab_size = logits.shape
    # perm = torch.randperm(vocab_size)
    # invperm = torch.argsort(perm, stable=True)

    # logits = logits[:, perm]

    # Now we want to, for each element in the batch, get the normalized
    # probabilities, stack them in the unit interval into buckets, and figure
    # out what bucket the code falls into.
    probs = F.softmax(logits, dim=1)

    # Use the numpy cumsum with host callback to guarantee nondecreasing array
    # of partial sums.
    cumprobs = torch.cumsum(probs, dim=1)

    # Because of precision, make sure the max value (and everything with that
    # value, to not change bucket widths) is at least 1.0.
    max_probs = torch.unsqueeze(torch.max(cumprobs, dim=1)[0], 1)
    all_bucket_maxes = torch.where((cumprobs == max_probs) & (cumprobs < 1.0), 1.0, cumprobs)

    # Now the cumulative probabilities represent the max value of each of the
    # buckets. So let's make a mask of all the buckets whose maxes are less
    # than and greater than the given codes.
    expanded_codes = torch.unsqueeze(codes, dim=1)
    bucket_maxes_lte_codes = all_bucket_maxes <= expanded_codes
    bucket_maxes_gt_codes = all_bucket_maxes > expanded_codes

    # Pick the minimum value for the bucket for the code. Note this will be
    # 0.0 if the code falls into the zero'th bucket, as desired.
    code_bucket_mins = torch.max(all_bucket_maxes * bucket_maxes_lte_codes, dim=1)[0]

    # We have to do some masking here, and for probabilities, anything > 1.0
    # is as good as infinity.
    prob_infty = 1.1
    # Pick the maximum value for the bucket, the first bucket whose max is
    # greater than the code.
    code_bucket_maxes = torch.min(
        all_bucket_maxes * bucket_maxes_gt_codes + bucket_maxes_lte_codes * prob_infty, dim=1
    )[0]
    # We have to take the argmin before inverting the permutation,
    # otherwise it messes up the default tie breaking behavior for size zero
    # buckets (take lowest index).
    sampled_indices_permed = torch.argmin(
        (all_bucket_maxes * bucket_maxes_gt_codes + bucket_maxes_lte_codes * prob_infty), dim=1
    )
    # sampled_indices = torch.argmax(
    #     F.one_hot(sampled_indices_permed, vocab_size)[:, invperm], dim=1)
    sampled_indices = torch.argmax(F.one_hot(sampled_indices_permed, vocab_size), dim=1)

    remainder_codes = (codes - code_bucket_mins) / (code_bucket_maxes - code_bucket_mins)

    samples = sampled_indices
    new_codes = remainder_codes

    return samples, new_codes


def make_default_codes(batch_size, num_decodes):
    """Make default codebook for a batch of `num_decodes` samples.

    The codes are initialized evenly spaced in the unit interval, with a random offset applied. This lets them evenly
    cover the sample space while also providing an unbiased estimate of any sample average.

    Args:
        batch_size: size of input batch.
        num_decodes: number of samples per batch element.

    Returns:
        [batch_size, num_decodes] array of codes.
    """
    offset = torch.rand((batch_size, 1))
    codes = torch.tile(
        torch.unsqueeze(torch.arange(1, num_decodes + 1, dtype=torch.float32) / (num_decodes + 1), axis=0),
        (batch_size, 1),
    )
    return torch.remainder(codes + offset, 1.0)
