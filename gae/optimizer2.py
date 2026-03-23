import torch
import torch.nn.modules.loss
import torch.nn.functional as F

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight, distance_matrix, x_hat=None, x=None,embed=None,pos_pairs=None, neg_pairs=None,margin=1.0,p=None,q=None, h=None, g=None):
    """
    Compute loss with adjacency reconstruction loss (MSE) and distance penalty.
    """
    # --- 确保所有张量都在同一设备上 ---
    device = preds.device
    labels = labels.to(device)
    mu = mu.to(device)
    logvar = logvar.to(device)
    distance_matrix = distance_matrix.to(device)
    if x_hat is not None:
        x_hat = x_hat.to(device)
    if x is not None:
        x = x.to(device)
    if embed is not None:
        embed = embed.to(device)







    # - preds: 预测的邻接矩阵 \(\hat{A}\)(torch.Tensor)[N, N]
    # - labels: 真实邻接矩阵 \(A\) (torch.Tensor) [N, N]
    #  - distance_matrix: 物理距离矩阵 \(d_{ij}\) (torch.Tensor) [N, N]
    # - alpha: 距离惩罚项的权重 (float)

    # L_recon: Adjacency reconstruction loss (MSELoss)
    #recon_loss = F.mse_loss(preds, labels)

    # L_dist: Distance penalty (only for connected nodes A_ij = 1)
    #distance_penalty = (distance_matrix * preds).sum()

    # Total loss
    #loss = recon_loss + alpha * distance_penalty
    
    # 先算未归一化的损失
    bce_loss = - (labels * torch.log(preds + 1e-15) + (1 - labels) * torch.log(1 - preds + 1e-15))
    
    distance_matrix = 1 / (1 + distance_matrix)  # 越近值越大，越远值越小
    
    #weighted_bce = distance_matrix * bce_loss
    # 按节点数平方归一化
    #recon_loss = weighted_bce.sum() / (n_nodes * n_nodes)
    
    bce = bce_loss.sum() / (n_nodes * n_nodes)  # unweighted BCE
    dist_bce = (distance_matrix * bce_loss).sum() / (n_nodes * n_nodes)  # distance-weighted BCE

    # ---- Final Reconstruction Loss: blended ----
    #recon_loss = (1 - alpha) * bce + alpha * dist_bce
    
    recon_loss =  q * dist_bce
    
    
    # KL Divergence
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    KLD_loss = g * KLD
    
    
    #print(f"recon_loss: {recon_loss.item():.4f}, KLD: {KLD.item():.4f}")


    #特征结构损失（MSE）
    if x_hat is not None and x is not None:
        feat_loss = F.mse_loss(x_hat, x)
        feat_loss = p * feat_loss
    else:
        feat_loss = None
    

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
        L_contrast_loss = h * L_contrast

    total_loss = recon_loss + KLD_loss + feat_loss + L_contrast_loss     
    
    #print(f"recon_loss: {recon_loss.item():.4f}, KLD_loss: {KLD_loss.item():.4f}, feat_loss:{feat_loss.item():.4f},L_contrast_loss:{L_contrast_loss.item():.4f},total_loss:{total_loss.item():.4f}")

    return total_loss,recon_loss,KLD_loss,feat_loss,L_contrast_loss
