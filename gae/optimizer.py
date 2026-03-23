import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight,x_hat=None, x=None, lambda_feat=1.0, beta=1.0,embed=None, pos_pairs=None, neg_pairs=None, contrastive_weight=0.1, margin=1.0):
    '''
    preds: reconstructed adjacency (after decoder activation if any; here we expect logits so use binary_cross_entropy_with_logits)
    labels: ground-truth adjacency labels (0/1)
    mu, logvar: encoder outputs (mu, logvar)  -- note: logvar here is assumed to be log(std) to match reparameterize
    n_nodes: number of nodes (N)
    norm, pos_weight: from getGraph
    x_hat, x: optional feature reconstruction and target
    lambda_feat: weight for feature loss
    beta: weight for KL term
    '''
    bce_loss = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight= pos_weight.to(preds.device))

    #特征重构损失（MSE），如果启用
    if x_hat is not None and x is not None:
        feat_loss = F.mse_loss(x_hat, x)
    else:
        feat_loss = None


    #KL 散度项：因为 reparameterize 使用 std = exp(logvar), 所以我们采用与你先前实现一致的公式
    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return bce_loss

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    #总损失
    total = bce_loss
    if feat_loss is not None:
        total = total + lambda_feat * feat_loss
    total = total + beta * KLD


    #对比损失
    if embed is not None and pos_pairs is not None and neg_pairs is not None:
        L_contrast = 0.0
        for i, j in pos_pairs:
            D = torch.norm(embed[i] - embed[j], p=2)
            L_contrast += D.pow(2)
        for i, j in neg_pairs:
            D = torch.norm(embed[i] - embed[j], p=2)
            L_contrast += F.relu(margin - D).pow(2)
        L_contrast = L_contrast / (len(pos_pairs) + len(neg_pairs) + 1e-6)

        #将对比损失加入总损失
        total = total + contrastive_weight * L_contrast


    return total, bce_loss.item(), (feat_loss.item() if feat_loss is not None else None), (KLD.item() if isinstance(KLD, torch.Tensor) else float(KLD))
