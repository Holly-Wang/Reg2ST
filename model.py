import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import anndata as ann

from performance import get_R
from attention import Decoder,Encoder
from wikg import WiKG

class LR_Scheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
    
def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor,float):
        scale_factor=np.full((len(mean),),scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor

    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(disp/(disp+mean+eps), disp)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda*torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result

def generate_masked_tensor(n, zero_prob=0.75):
    matrix = torch.ones(n, n)
    rand_tensor = torch.rand(n, n)
    mask = (rand_tensor < zero_prob) & (torch.eye(n) == 0)
    matrix[mask] = 0    
    return matrix

class T_SIMSIAM(pl.LightningModule):
    def __init__(self, args):
        super(T_SIMSIAM, self).__init__()
        self.args = args
        dim_in = args.dim_in
        dim_hidden = args.dim_hidden
        dim_out = args.dim_out
        dropout = args.dropout
        wikg_top = args.wikg_top
        decoder_layer = args.decoder_layer
        decoder_head = args.decoder_head
        self.model_name = "clip_pretrain"
        self.image_encoder = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_hidden)
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.gene_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_out)
        )
        self.decoder = Decoder(dim=dim_hidden, layers=decoder_layer, heads=decoder_head, mlp_dim=1024, dropout=dropout)

        self.embed_x = nn.Embedding(64, dim_hidden)
        self.embed_y = nn.Embedding(64, dim_hidden)
        self.gene_linear = nn.Sequential(
            nn.Linear(dim_out, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, dim_hidden)
            )
        self.mean = nn.Sequential(nn.Linear(dim_hidden, dim_out), MeanAct())
        self.disp = nn.Sequential(nn.Linear(dim_hidden, dim_out), DispAct())
        self.pi = nn.Sequential(nn.Linear(dim_hidden, dim_out), nn.Sigmoid())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.lr_scheduler = None
        self.graph_layer = WiKG(dim_in=dim_hidden, dim_hidden=dim_hidden, topk=wikg_top, n_classes=dim_hidden)


    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def encode_image(self, image):
        x = self.image_encoder(image)
        proj = self.gene_proj(x)
        return x, proj

    def encode_gene(self, gene):
        x = self.gene_linear(gene)
        return x


    def forward(self, gene, image, pos):
        i_f, proj_i_f = self.encode_image(image)
        # print(pos[:, 0])
        x = self.embed_x(pos[:, 0])
        y = self.embed_y(pos[:, 1])
        i_ct = i_f.clone() + x + y
        # bake_loss = F.mse_loss(i_f, bake_i_f)
        g_f = self.encode_gene(gene)
        g_f = g_f / g_f.norm(dim=1, keepdim=True)
        # i_g = self.gene_head(i_f)
        # g_g = self.gene_head(g_f)
        proj_loss = F.mse_loss(g_f, proj_i_f)
        if self.training:
            mask = generate_masked_tensor(g_f.shape[0])
            i_g = self.decoder(i_ct, proj_i_f,mask.cuda())
        else:
            i_g = self.decoder(i_ct, proj_i_f)
        i_g = self.decoder(i_ct, proj_i_f)  
        i_g = self.graph_layer(torch.unsqueeze(i_g, dim=0), torch.unsqueeze(i_ct.clone(), dim=0), torch.unsqueeze(proj_i_f.clone(), dim=0)).squeeze(dim=0)
        m = self.mean(i_g)
        d = self.disp(i_g)
        p = self.pi(i_g)
        extra = m, d, p
        i_g = self.gene_head(i_g)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * i_f @ g_f.t()
        logits_per_gene = logits_per_image.t()
        return logits_per_image, logits_per_gene, i_g, i_g, extra, proj_loss
        
    def forward_clip(self, gene, image):
        # print(image.shape, gene.shape)
        image_features = self.encode_image(image)
        gene_features = self.encode_gene(gene)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        gene_features = gene_features / gene_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ gene_features.t()
        # print("logits per image: ", logits_per_image)
        logits_per_gene = logits_per_image.t()        

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_gene, image_features
   
    def infoNCE_loss(self, logits_per_image, logits_per_gene):
        batch_size = logits_per_image.shape[0]
        labels = torch.eye(batch_size, device=logits_per_image.device)
        loss_i = F.binary_cross_entropy_with_logits(logits_per_image, labels)
        loss_g = F.binary_cross_entropy_with_logits(logits_per_gene, labels)
        loss = (loss_i + loss_g) / 2.0
        return loss
    def training_step(self, batch, batch_idx):
        g, i, pos, _, oris, sfs = batch
        g = torch.squeeze(g)
        i = torch.squeeze(i)
        pos = torch.squeeze(pos)
        logits_per_image, logits_per_gene, i_g, g_g, extra, proj_loss = self.forward(g, i, pos)
        m,d,p=extra
        zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0)) * 0.25
        info_loss = self.infoNCE_loss(logits_per_image, logits_per_gene) * 0.5
        mse_loss = F.mse_loss(g, i_g)
        train_loss = info_loss + mse_loss + proj_loss + zinb_loss
        self.log('zinb_loss', zinb_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('con_loss', info_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('proj_loss', proj_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return train_loss
    def validation_step(self, batch, batch_idx):
        # 和 training_step 类似地处理 batch 数据
        g, i, pos, _, _, _ = batch
        g = torch.squeeze(g)
        i = torch.squeeze(i)
        pos = torch.squeeze(pos)
        _, _, i_g, _, _, _ = self.forward(g, i, pos)
        p, r = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        g, i, pos, centers, _, _ = batch
        g = torch.squeeze(g)
        i = torch.squeeze(i)
        pos = torch.squeeze(pos)
        centers = torch.squeeze(centers)
        _, _, i_g, _, _, _ = self.forward(g, i, pos)
        adata = ann.AnnData(X=i_g.detach().cpu().numpy())
        adata.obsm["spatial"] = centers.detach().cpu().numpy()
        p, r = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)
        self.p = p
        self.r = r
        self.data = adata
  
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        self.lr_scheduler = LR_Scheduler(self.optimizer, 10, 1e-5, self.args.epochs, 1e-4, 1e-6, 31)
        return self.optimizer
        
