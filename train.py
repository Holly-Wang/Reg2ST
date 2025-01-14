from unittest import TestLoader
from venv import logger

from regex import F
from dlpfc import DLPFCHandler
from utils import parser_option
from pretrain_model import Clip_Pretrain
from herst import ViT_HER2ST, ViT_SKIN
# from MODEL.utils.performance import get_R
from model import T_SIMSIAM
from pytorch_lightning.loggers import CSVLogger
import torch
import numpy as np
import os


from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from performance import get_R

def train(args):
    torch.set_float32_matmul_precision('high')
    i = args.fold
    args.divide_size = 1
    save_dir = f'./model_wikg{args.wikg_top}_cross{args.decoder_layer}_{args.decoder_head}_mask{args.mask_rate}_zinb{args.w_zinb}_con{args.w_con}_{args.dim_in}_{args.dim_hidden}_{args.dataset}_{args.epochs}_fold{i}/'
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="model-{epoch}-{pcc}",
        save_last=False,
        save_top_k=-1,
        monitor='pcc',
        mode='max',
        every_n_epochs=50
    )
    max_pcc_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="model-{epoch}-{pcc}",
        save_last=False,
        save_top_k=1,
        monitor='pcc',
        mode='max',
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

    model = T_SIMSIAM(args=args)
    trainer = pl.Trainer(logger=csv_logger, precision=32, max_epochs=args.epochs, 
                         accelerator='gpu', devices=[args.device_id], 
                         callbacks=[val_checkpoint_callback, max_pcc_checkpoint_callback], 
                         log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)
    
    # testdata = ViT_HER2ST(args, train=False, flatten=False,ori=True, adj=True, prune='Grid', r=4, fold=i, val=False)
    
    # ddp_trainer = pl.Trainer(
    # fast_dev_run=False,
    # max_epochs=400,accelerator="gpu", devices=1,
    # precision=32,log_every_n_steps=1,
    # check_val_every_n_epoch=1,
    # val_check_interval=1.0,
    # num_sanity_val_steps=0,
    # )
    # ddp_trainer.test(model,valloader,verbose=True,ckpt_path=val_checkpoint_callback.best_model_path)
    # trainer = pl.Trainer(max_epochs=400, accelerator='gpu', devices=1, check_val_every_n_epoch=1, 
    #                          callbacks=[val_checkpoint_callback]
    #                          )
    # trainer.test(model, testloader)
        # trainer.test(model, testloader)


def predict(args):
    pcc = []
    r = []
    torch.set_float32_matmul_precision('high')
    # if args.dataset == "her2st":
    #     for i in range(32):
    #         val_data = ViT_HER2ST(train=False, flatten=False,ori=True, adj=False, fold=i)
    # else:
    #     for i in range(12):
    #         args.dim_out = 171
    #         val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=i)
    args.dim_hidden = 256
    args.dim_out = 171
    args.wikg_top=6
    args.decoder_layer=12
    args.decoder_head=8
    for i in range(12):
        val_data = ViT_SKIN(train=False, flatten=False,ori=True, adj=False, fold=i)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
        ckpt_dir = f"model_wikg6_cross12_8_mask0.75_zinb0.25_con0.5_1024_256_cscc_700_fold{i}/"
        ckpt_files = [
            os.path.join(ckpt_dir, file)
            for file in os.listdir(ckpt_dir)
            if file.startswith("model-epoch=399-pcc") and file.endswith(".ckpt")
        ]
        print(ckpt_files)
        model = T_SIMSIAM(args)
        trainer = pl.Trainer(precision=32, max_epochs=args.epochs, 
                         accelerator='gpu', devices=[args.device_id])
        trainer.test(model, val_loader, ckpt_path=ckpt_files[0])
        pcc.append(model.p)
        r.append(model.r)

    np.save("skin_pcc_399_final", pcc)
    np.save("skin_r_399_final", r)        
def save_adata(args):
    torch.set_float32_matmul_precision('high')
    args.dim_hidden = 256
    args.dim_out = 785
    args.wikg_top=6
    args.decoder_layer=6
    args.decoder_head=8
    for i in [5, 23]:
        val_data = ViT_HER2ST(train=False, flatten=False,ori=True, adj=False, fold=i)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
        ckpt_dir = f"model_wikg6_cross6_8_mask0.75_zinb0.25_con0.5_1024_256_her2st_400_fold{i}_v1/"
        ckpt_files = [
            os.path.join(ckpt_dir, file)
            for file in os.listdir(ckpt_dir)
            if file.startswith("model-epoch=399-pcc") and file.endswith(".ckpt")
        ]
        print(ckpt_files)
        model = T_SIMSIAM(args)
        trainer = pl.Trainer(precision=32, max_epochs=args.epochs, 
                         accelerator='gpu', devices=[args.device_id])
        trainer.test(model, val_loader, ckpt_path=ckpt_files[0])
        data = model.data
        data.write(f"her2st_adata/{i}_v1.h5ad")
       
        
if __name__ == "__main__":
    args = parser_option()
    # train(args)
    predict(args)
    # save_adata(args)
    