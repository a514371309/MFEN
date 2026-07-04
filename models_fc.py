"""
TMC variant: pretrained encoders + FC decision heads (no fuzzy TSK).
Evidence fusion (Dempster-Shafer) is kept the same as demo/models.py TMC.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import BertClf, ImageEncoder, ce_loss, sim_loss


class FCHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.net(x)


class TMC_FC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.bert = BertClf(args)

        bert_dim = args.hidden_sz_bert
        rgb_dim = args.img_hidden_sz * args.num_image_embeds
        hidden = getattr(args, "fc_hidden", 256)
        dropout = getattr(args, "dropout", 0.1)

        self.clf_bert_fc = FCHead(bert_dim, args.n_classes, hidden_dim=hidden, dropout=dropout)
        self.clf_rgb_fc = FCHead(rgb_dim, args.n_classes, hidden_dim=hidden, dropout=dropout)

    def DS_Combin_two(self, alpha1, alpha2):
        alpha = {0: alpha1, 1: alpha2}
        b, S, E, u = {}, {}, {}, {}
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)

        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        S_a = self.args.n_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        return e_a + 1

    def forward(self, batch, device):
        txt, segment, mask, img, tgt, idx = batch
        img = img.to(device)
        txt, mask, segment = txt.to(device), mask.to(device), segment.to(device)

        bert_feat = self.bert(txt, mask, segment)
        rgb_feat = self.rgbenc(img)
        rgb_feat = torch.flatten(rgb_feat, start_dim=1)

        depth_out = self.clf_bert_fc(bert_feat)
        rgb_out = self.clf_rgb_fc(rgb_feat)

        depth_evidence = F.softplus(depth_out)
        rgb_evidence = F.softplus(rgb_out)
        depth_alpha = depth_evidence + 1
        rgb_alpha = rgb_evidence + 1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha, depth_out, rgb_out
