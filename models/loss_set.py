from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import kl_div, softmax, log_softmax


# KL Divergence Loss
class KLDivLoss(nn.Module):
    def __init__(self, temperature):
        super(KLDivLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb1, emb2):
        emb1 = softmax(emb1 / self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2 / self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv


# Ranking Loss
class RankingLoss(nn.Module):
    def __init__(self, neg_penalty):
        super(RankingLoss, self).__init__()
        self.neg_penalty = neg_penalty

    def forward(self, ranks, labels, class_ids_loaded, device):
        """
        For each correct rank, it should be higher than the absence rank.
        """
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1 + (labels * -1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i] == 1]
            wrong = ranks_loaded[i, neg_labels[i] == 1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty + wrong) - correct)
            image_level_penalty[image_level_penalty < 0] = 0
            loss_rank += image_level_penalty.sum()
        loss_rank /= len(labels)

        return loss_rank


# Cosine Similarity Loss
class CosineLoss(nn.Module):
    def forward(self, cxr, ehr):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        return loss


# General Loss Class
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        # Extract loss name and parameters from args
        loss_name = args.loss.lower()  # Expect `args` to have a `loss_name` attribute
        self.loss_name = loss_name

        if self.loss_name == 'kldiv':
            self.loss_fn = KLDivLoss(temperature=args.temperature)
        elif self.loss_name == 'ranking':
            self.loss_fn = RankingLoss(neg_penalty=args.neg_penalty)
        elif self.loss_name == 'cosine':
            self.loss_fn = CosineLoss()
        elif self.loss_name == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)
