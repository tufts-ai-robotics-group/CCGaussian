import torch

import ccgaussian.loss as loss


batch_size = 8
embed_dim = 768
num_classes = 5
num_batches = 3


def test_unsup_nll():
    unsup_loss = loss.UnsupNLLLoss(num_classes, num_batches)
    embeds = torch.randn((batch_size, embed_dim))
    means = torch.randn((num_classes, embed_dim))
    sigma2s = torch.ones((embed_dim,))
    for _ in range(num_batches):
        assert unsup_loss(embeds, means, sigma2s) == 0
    assert unsup_loss(embeds, means, sigma2s) != 0
