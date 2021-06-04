import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VATLoss(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model):
        super(VATLoss, self).__init__()
        self.model = model
        self.kl_div = nn.KLDivLoss(size_average=False, reduce=False).to(device)

    def forward(self, X , logit):
        vat_loss = virtual_adversarial_loss(X, logit , self.model, self.kl_div)
        return vat_loss  # already averaged


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp


def get_normalized_vector(d):
    return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

def generate_virtual_adversarial_perturbation(x, model):
    d = torch.randn_like(x)
    return get_normalized_vector(d).requires_grad_()


def virtual_adversarial_loss(x,logit, model , kl_div):
    r_vadv = generate_virtual_adversarial_perturbation(x, model)
    prob_logits = F.softmax(logit.detach(), dim=1)
    logit_m = model(x + r_vadv)
    loss = torch.mean(kl_div(
            F.log_softmax(logit_m, dim=1), prob_logits).sum(dim=1))
    return loss