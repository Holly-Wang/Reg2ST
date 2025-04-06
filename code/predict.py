from herst import ViT_HER2ST, ViT_SKIN
from model import Reg2ST
from pytorch_lightning.loggers import CSVLogger
import torch
import numpy as np
import os


from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from performance import get_R
from utils import *
    
def predict(args):
    torch.set_float32_matmul_precision('high')
    if args.dataset == "her2st":
        val_data = ViT_HER2ST(train=False, flatten=False,ori=True, adj=False, fold=i)
        args.dimout = 785
    else:
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=i)
        args.dim_out = 171
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
    ckpt_dir = f"{args.dataset}_model/fold{i}_model.ckpt"
    print(f":oading checkpoints from {ckpt_dir}")
    model = Reg2ST(args)
    trainer = pl.Trainer(precision=32, max_epochs=args.epochs, 
                         accelerator='gpu', devices=[args.device_id])
    trainer.test(model, val_loader, ckpt_path=ckpt_dir)
        
if __name__ == "__main__":
    args = parser_option()
    predict(args)
    