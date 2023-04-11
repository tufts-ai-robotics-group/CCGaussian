import torch
import torch.nn
import torch.nn.functional as f


class NDCCLoss(torch.nn.Module):
    def __init__(self, w_nll) -> None:
        super().__init__()
        self.w_nll = w_nll

    @staticmethod
    def ce_loss(logits, targets):
        return f.cross_entropy(logits, targets)

    @staticmethod
    def sq_mahalanobis_d(embeds, means, sigma2s, targets):
        # goes from B x D to B
        return ((embeds - means[targets])**2 / sigma2s).sum(dim=1)

    @staticmethod
    def nll_loss(embeds, means, sigma2s, targets):
        # negative log-likelihood loss
        return torch.log(sigma2s).sum() / 2 + \
            torch.mean(NDCCLoss.sq_mahalanobis_d(embeds, means, sigma2s, targets)) / 2

    def forward(self, logits, embeds, means, sigma2s, targets):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + \
            self.w_nll * self.nll_loss(embeds, means, sigma2s, targets)


class NDCCFixedLoss(NDCCLoss):
    # NDCCLoss for fixed variance
    def forward(self, logits, embeds, means, sigma2s, targets):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        return self.ce_loss(logits, targets) + self.w_nll * \
            torch.mean(NDCCLoss.sq_mahalanobis_d(embeds, means, sigma2s, targets)) / 2


class NDCCFixedSoftLoss(NDCCLoss):
    def __init__(self, w_nll, w_novel) -> None:
        super().__init__(w_nll)
        self.w_novel = w_novel

    # NDCCLoss for soft labels and fixed variance
    def forward(self, logits, embeds, means, sigma2s, soft_targets, norm_mask):
        if embeds.shape[0] == 0:
            return torch.scalar_tensor(0.)
        # validate soft_targets
        assert torch.allclose(torch.sum(soft_targets, axis=1),
                              torch.ones(soft_targets.shape[0]).to(soft_targets.device))
        # take sum over clusters then later mean over batch dimension to get scalar
        md_reg = torch.sum(all_sq_md(embeds, means, sigma2s) * soft_targets, dim=1) / 2
        # normal loss, checking for empty inputs
        if norm_mask.sum() > 0:
            norm_l = self.ce_loss(logits[norm_mask], soft_targets[norm_mask]) + \
                self.w_nll * md_reg[norm_mask].mean()
        else:
            norm_l = 0
        # novel loss, checking for empty inputs
        if (~norm_mask).sum() > 0:
            novel_l = self.ce_loss(logits[~norm_mask], soft_targets[~norm_mask]) + \
                self.w_nll * md_reg[~norm_mask].mean()
        else:
            novel_l = 0
        return (1 - self.w_novel) * norm_l + self.w_novel * novel_l


def all_sq_md(embeds, means, sigma2s):
    # Mahalanobis distance for embeddings to each class
    # goes from B x K x D to B x K
    return torch.sum((torch.unsqueeze(embeds, dim=1) - means)**2 / sigma2s, dim=2)


def novelty_sq_md(embeds, means, sigma2s):
    # Mahalanobis distance for embeddings to closest class
    # goes from B x K to B
    return torch.min(all_sq_md(embeds, means, sigma2s), dim=1)[0]
