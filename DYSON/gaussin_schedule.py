import random

import numpy as np
import math

import torch


def prob2int(prob):
    x = torch.rand(1)[0]
    if x < prob:
        return 1
    else:
        return 0


def gaussian(peak, width, x):
    out = math.exp(- ((x - peak) ** 2 / (2 * width ** 2)))
    return out


def split_way_schedule(num_way, num_classes, bs, data_len):
    random.seed(1993)
    labels = [x for x in range(num_classes)]
    random.shuffle(labels)
    labels = torch.tensor(labels)
    labels = labels.view(num_way, -1)
    full_batches = data_len / bs
    schedule = [{'label_set': lbl.tolist(), 'n_batches': full_batches/num_way} for lbl in labels]
    return schedule


def gaussian_schedule(num_classes):
    """Returns a schedule where one task blends smoothly into the next."""

    schedule_length = 1000  # schedule length in batches
    episode_length = 5  # episode length in batches

    # Each class label appears according to a Gaussian probability distribution
    # with peaks spread evenly over the schedule
    peak_every = schedule_length // num_classes
    width = 50  # width of Gaussian
    peaks = range(peak_every // 2, schedule_length, peak_every)

    schedule = []
    labels = np.arange(num_classes)
    np.random.seed(1993)
    np.random.shuffle(labels)  # labels in random order

    for ep_no in range(0, schedule_length // episode_length):

        lbls = []
        while lbls == []:  # make sure lbls isn't empty
            for j in range(len(peaks)):
                peak = peaks[j]
                # Sample from a Gaussian with peak in the right place
                p = gaussian(peak, width, ep_no * episode_length)
                add = prob2int(p)
                if add:
                    lbls.append(int(labels[j]))

        episode = {'label_set': lbls, 'n_batches': episode_length}
        schedule.append(episode)

    return schedule
