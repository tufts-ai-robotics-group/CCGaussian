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
        """Unsupervised GMM NLL loss

        Args:
            num_classes (int): Number of GMM clusters K
            num_batches (int): Number of batches per epoch for max queue size
        """
        super().__init__()
        # queue for calculating mixing coefficients, up to N x K
        self.log_resp_queue = torch.empty(0, num_classes)
        # mixing coefficient used for initialization epoch
        self.init_mix_coef = torch.log(torch.Tensor([1 / num_classes] * num_classes))
        # current mixing coefficients, size K
        self.log_mix_coef = self.init_mix_coef
        # max queue size
        self.max_batches = num_batches
        # number of batches queued currently
        self.cur_batches = 0

    def all_ll(self, embeds, means, sigma2s):
        # GMM log likelihood for each cluster, output B x K
        cc_ll = -torch.log(2 * torch.scalar_tensor(torch.pi)) * (len(sigma2s) / 2) \
            - torch.log(sigma2s).sum() / 2 \
            - all_mahalanobis(embeds, means, sigma2s) / 2
        # use learned mixing coef if queue has been filled
        if self.cur_batches == self.max_batches:
            log_mix_coef = self.log_mix_coef
        else:
            log_mix_coef = self.init_mix_coef
        return log_mix_coef + cc_ll

    def update_resp(self, batch_log_resp):
        # update mixing coefficient and responsibility queue
        # enqueue batch of responsiblities B x K
        self.log_resp_queue = torch.vstack((self.log_resp_queue, batch_log_resp))
        self.cur_batches += 1
        # dequeue if full
        if self.cur_batches > self.max_batches:
            self.log_resp_queue = self.log_resp_queue[batch_log_resp.shape[0]:]
            self.cur_batches -= 1
        # update mixing coefficient, mean of the queued entries in log space
        self.log_mix_coef = torch.logsumexp(self.log_resp_queue, dim=0) \
            - torch.log(torch.scalar_tensor(self.log_resp_queue.shape[0]))

    def forward(self, embeds, means, sigma2s):
        """Unsupervised GMM NLL loss

        Args:
            embeds (torch.Tensor): Latent space embeddings (B, D)
            means (torch.Tensor): Cluster means (K, D)
            sigma2s (torch.Tensor): Cluster independent covariance diagonal (D,)

        Returns:
            float: Loss value
        """
        all_ll = self.all_ll(embeds, means, sigma2s)
        # log likelihood of each input, B x 1
        total_ll = torch.logsumexp(all_ll, dim=1, keepdim=True)
        # cluster log responsibility for each input, B x K
        batch_log_resp = all_ll - total_ll
        # check queue size before updating
        filling_queue = self.cur_batches < self.max_batches
        self.update_resp(batch_log_resp)
        # stop after updating queue until initialized from full epoch
        if filling_queue:
            return torch.scalar_tensor(0)
        return -total_ll.sum()
