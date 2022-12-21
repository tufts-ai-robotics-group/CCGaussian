import torch
import torch.nn
import torch.nn.functional as f


def all_mahalanobis(embeds, means, sigma2s):
    # B x K x D to B x K
    return torch.sum((torch.unsqueeze(embeds, dim=1) - means)**2 / sigma2s, dim=2)


class NDCCLoss(torch.nn.Module):
    def __init__(self, w_nll) -> None:
        super().__init__()
        self.w_nll = w_nll

    @staticmethod
    def ce_loss(logits, targets):
        return f.cross_entropy(logits, targets)

    @staticmethod
    def cc_mahalanobis(embeds, means, sigma2s, targets):
        # class conditional Mahalanobis distance
        return ((embeds - means[targets])**2 / sigma2s).sum()

    @staticmethod
    def nll_loss(embeds, means, sigma2s, targets):
        # negative log-likelihood loss
        return torch.log(sigma2s).sum() / 2 + \
            NDCCLoss.cc_mahalanobis(embeds, means, sigma2s, targets) / \
            (2 * embeds.shape[0])

    def forward(self, logits, embeds, means, sigma2s, targets):
        return self.ce_loss(logits, targets) + \
            self.w_nll * self.nll_loss(embeds, means, sigma2s, targets)


class UnsupNLLLoss(torch.nn.Module):
    def __init__(self, num_classes, num_batches) -> None:
        super().__init__()
        # queue for calculating mixing coefficients, up to num_batches x K
        self.log_resp_queue = torch.Tensor([[]])
        # mixing coefficient used for initialization epoch
        self.init_mix_coef = torch.log(torch.Tensor([1 / num_classes] * num_classes))
        # current mixing coefficient, K
        self.log_mix_coef = self.init_mix_coef
        # number of batches per epoch, max queue size
        self.num_batches = num_batches

    def all_ll(self, embeds, means, sigma2s):
        # GMM log likelihood for each cluster, B x K
        cc_ll = -torch.log(2 * torch.pi) * (len(sigma2s) / 2) \
            - torch.log(sigma2s).sum() / 2 \
            - all_mahalanobis(embeds, means, sigma2s) / 2
        if self.log_resp_queue.shape[0] == self.num_batches:
            log_mix_coef = self.log_mix_coef
        else:
            log_mix_coef = self.init_mix_coef
        return log_mix_coef + cc_ll

    def batch_log_resp(self, all_ll):
        # B x K to B x K
        return all_ll - torch.logsumexp(all_ll, dim=1)

    def update_resp(self, batch_log_resp):
        batch_len = batch_log_resp.shape[0]
        # enqueue
        log_resp_sum = torch.logsumexp(batch_log_resp, dim=0) - torch.log(batch_len)
        self.log_resp_queue = torch.vstack((self.log_resp_queue, log_resp_sum))
        self.log_mix_coef += log_resp_sum
        # dequeue if full
        if self.log_resp_queue.shape[0] > self.num_batches:
            self.log_mix_coef -= self.log_resp_queue[0]
            self.log_resp_queue = self.log_resp_queue[1:]

    def forward(self, embeds, means, sigma2s):
        all_ll = self.all_ll(embeds, means, sigma2s)
        batch_log_resp = self.batch_log_resp(all_ll)
        self.update_resp(batch_log_resp)
        # only enqueue responsibilities until initialized from full epoch
        if len(self.log_resp_queue) < self.num_batches:
            return 0
        return -torch.logsumexp(all_ll, dim=1).sum()
