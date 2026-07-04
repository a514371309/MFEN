import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tsk import AntecedentGMF,TSK
from pytorch_pretrained_bert.modeling import BertModel
# from transformers import BertTokenizer

class GloveBowEncoder(nn.Module):
    def __init__(self, args):
        super(GloveBowEncoder, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_sz, args.embed_sz)
        self.load_glove()
        self.embed.weight.requires_grad = False

    def load_glove(self):
        print("Loading glove")
        pretrained_embeds = np.zeros(
            (self.args.vocab_sz, self.args.embed_sz), dtype=np.float32
        )
        for line in open(self.args.glove_path):
            w, v = line.split(" ", 1)
            if w in self.args.vocab.stoi:
                pretrained_embeds[self.args.vocab.stoi[w]] = np.array(
                    [float(x) for x in v.split()], dtype=np.float32
                )
        self.embed.weight = torch.nn.Parameter(torch.from_numpy(pretrained_embeds))

    def forward(self, x):
        return self.embed(x).sum(1)


class GloveBowClf(nn.Module):
    def __init__(self, args):
        super(GloveBowClf, self).__init__()
        self.args = args
        self.enc = GloveBowEncoder(args)
        self.clf = nn.Linear(args.embed_sz, args.n_classes)

    def forward(self, x):
        return self.clf(self.enc(x))

class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        _, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return out


class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        #self.clf = nn.Linear(args.hidden_sz_bert, args.n_classes)
        #self.clf.apply(self.enc.bert.init_bert_weights)

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        return x#self.clf(x)
class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        model = torchvision.models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048
# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl
def sim_loss(alpha_rgb,alpha_depth,p):
    loss =torch.mean( p*(alpha_rgb-alpha_depth)**2)
    return loss

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))


class TMC(nn.Module):
    def __init__(self, args):
        super(TMC, self).__init__()
        self.args = args
        self.rgbenc = ImageEncoder(args)
        self.bert = BertClf(args)
        #self.depthenc = ImageEncoder(args)
        depth_last_size =args.hidden_sz_bert# args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        #self.clf_depth = nn.ModuleList()
        #self.clf_rgb = nn.ModuleList()
        self.clf_bert=nn.Sequential(AntecedentGMF(in_dim=depth_last_size, n_rule=args.n_rule, high_dim=True, init_center=None),
                                     nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
       
        self.clf_rgb = nn.Sequential(AntecedentGMF(in_dim=rgb_last_size, n_rule=args.n_rule, high_dim=True, init_center=None),
                                    nn.LayerNorm(args.n_rule),
                                    nn.ReLU())
        self.clf_bert_tsk=TSK(in_dim=depth_last_size, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.clf_bert, order=args.order, precons=None)
        self.clf_rgb_tsk=TSK(in_dim=rgb_last_size, out_dim=args.n_classes, n_rule=args.n_rule, antecedent=self.clf_rgb, order=args.order, precons=None)
        



    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.args.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, batch,device):
        
        txt, segment,mask,  img, tgt,idx = batch
        #print(img)
        img = img.to(device)
        txt, mask, segment = txt.to(device), mask.to(device), segment.to(device)
        #out = model(txt, mask, segment)
        depth = self.bert(txt, mask, segment)
        #depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(img)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = self.clf_bert_tsk(depth)
        rgb_out = self.clf_rgb_tsk(rgb)
        

        depth_evidence, rgb_evidence = F.softplus(depth_out), F.softplus(rgb_out)
        depth_alpha, rgb_alpha = depth_evidence+1, rgb_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha,depth_out,rgb_out


class ETMC(TMC):
    def __init__(self, args):
        super(ETMC, self).__init__(args)
        last_size = args.img_hidden_sz * args.num_image_embeds + args.img_hidden_sz * args.num_image_embeds
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden
        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, rgb, depth):
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        pseudo_out = torch.cat([rgb, depth], -1)
        for layer in self.clf:
            pseudo_out = layer(pseudo_out)

        depth_evidence, rgb_evidence, pseudo_evidence = F.softplus(depth_out), F.softplus(rgb_out), F.softplus(pseudo_out)
        depth_alpha, rgb_alpha, pseudo_alpha = depth_evidence+1, rgb_evidence+1, pseudo_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha