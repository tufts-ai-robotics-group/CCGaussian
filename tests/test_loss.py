import pytest
import torch

import ccgaussian.loss


batch_size = 8
embed_dim = 2
num_classes = 2
num_batches = 5


class TestUnsupNLL():
    @pytest.fixture
    def loss(self):
        return ccgaussian.loss.UnsupNLLLoss(num_classes, num_batches)

    def test_queue_delay(self, loss):
        embeds = torch.randn((batch_size, embed_dim))
        means = torch.randn((num_classes, embed_dim))
        sigma2s = torch.ones((embed_dim,))
        for _ in range(num_batches):
            assert loss(embeds, means, sigma2s) == 0
        assert loss(embeds, means, sigma2s) != 0

    def test_nll(self, loss):
        embeds = torch.Tensor([[0, 0], [1, 0], [-1, 0]])
        means = torch.Tensor([[1, 0], [-1, 0]])
        sigma2s = torch.ones((embed_dim,))
        for _ in range(num_batches):
            loss(embeds, means, sigma2s)
        all_ll = loss.all_ll(embeds, means, sigma2s)
        total_ll = torch.logsumexp(all_ll, dim=1)
        # construct true mixture
        true_mix = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(torch.ones(2)),
            torch.distributions.MultivariateNormal(means, torch.diag(sigma2s)))
        assert torch.allclose(true_mix.log_prob(embeds), total_ll)
