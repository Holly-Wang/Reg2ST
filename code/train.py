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

def train(args):
    torch.set_float32_matmul_precision('high')
    i = args.fold
    args.divide_size = 1
    save_dir = f"./{args.dataset}_model/"
    try:
        os.mkdir(f"./{args.dataset}_model/")
    except:
        pass
    
    val_checkpoint_callback = ModelCheckpoint(
    dirpath=save_dir,
    filename=f"fold{i}_model",
    save_last=True,
    save_top_k=0,
    save_on_train_epoch_end=True
    )
    logger_name = f'{args.dataset}_fold{i}/'
    csv_logger = CSVLogger(logger_name, name=f"fold{i}")
    if args.dataset == "her2st":
        train_data = ViT_HER2ST(train=True, flatten=False,ori=True, adj=False, fold=i)
        val_data = ViT_HER2ST(train=False, flatten=False,ori=True, adj=False, fold=i)
    else:
        args.dim_out = 171
        train_data = ViT_SKIN(train=True, flatten=False, ori=True, adj=False, fold=i)
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=i)
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    model = Reg2ST(args=args)
    trainer = pl.Trainer(logger=csv_logger, precision=32, max_epochs=args.epochs, 
                         accelerator='gpu', devices=[args.device_id], 
                         callbacks=[val_checkpoint_callback], 
                         log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)
       
        
if __name__ == "__main__":
    args = parser_option()
    train(args)
    