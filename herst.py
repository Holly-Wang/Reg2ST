
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance
from collections import defaultdict as dfd
from copy import deepcopy
import os
import torch
import pandas as pd
import scanpy as sc
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from collections import defaultdict
from torchvision import transforms
import glob
import random
from torch.utils.data import DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def split_list(seed=None):
    if seed is not None:
        random.seed(seed)
    data = list(range(31))
    train_size = len(data) * 8 // 9  # 8:1 划分
    test_size = len(data) - train_size

    train_set = random.sample(data, train_size)

    test_set = [item for item in data if item not in train_set]

    return train_set, test_set


def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix=coord#.cpu().numpy()
    nodes=spatialMatrix.shape[0]
    Adj=torch.zeros((nodes,nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    return Adj

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train: True, fold=1,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4, val=True):
        super(ViT_HER2ST, self).__init__()
        
        self.cnt_dir = '/d/wangx/clip_prediction/data/her2st/ST-cnts'
        self.img_dir = '/d/wangx/clip_prediction/data/her2st/ST-imgs'
        self.pos_dir = '/d/wangx/clip_prediction/data/her2st/ST-spotfiles'
        self.lbl_dir = '/d/wangx/clip_prediction/data/her2st/ST-pat/lbl'
        self.r = 224//2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('/d/wangx/clip_prediction/data/her2st/her_hvg_cut_1000.npy',allow_pickle=True))
        
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)

        names = [i[:2] for i in names]
        self.train = train
        self.val = val
        self.ori = ori
        self.adj = adj
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        names.sort()
        samples = names[1:33]

        te_names = [samples[fold]]
        print("test sample:", te_names)
        tr_names = list(set(samples) - set(te_names))
        
        if self.train:
            self.names = tr_names
        else:
            self.names = te_names
        
        
        self.names.sort()
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not self.train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif self.train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    lbl = torch.Tensor([self.lbl2id[i] for i in lbl])
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i,m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict = defaultdict(type(None))
        # self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        self.patch_path = '/d/wangx/phikonv2_embedding/her2st/'
            
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        # adj = self.adj_dict[ID]
        # patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        img_f = np.load(f"{self.patch_path}{ID}.npy")
        data = [torch.Tensor(exps), img_f, positions, torch.Tensor(centers)]
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        # print(name)
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df


class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=2,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_SKIN, self).__init__()

        self.dir = '/d/wangx/clip_prediction/data/GSE144240_RAW/'
        self.r = 224//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load(f'{self.dir}skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        self.names.sort()
        # print(self.names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.patch_path = '/d/wangx/phikonv2_embedding/skin/'


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        # im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        # patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)

        img_f = np.load(f'{self.patch_path}{ID}.npy')
        data=[torch.Tensor(exps), img_f, positions, torch.Tensor(centers)]
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

def cut_pic():
    data = ViT_SKIN(train=True, fold=0, flatten=False, adj=False, ori=False)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True)
    i=1
    for data in dataloader:
        # _, _, pic, _, _, _ = data
        _, _, _, pic  = data
        print(pic.shape)
        np.save(f"/d/wangx/patches/skin/{i}.npy", pic.numpy())
        i=i+1
    data = ViT_SKIN(train=False, fold=0, flatten=False, ori=False, adj=False)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True)
    for data in dataloader:
        # _, _, pic, _, _, _ = data
        # pic, _, _, _ = data
        _, _, _, pic = data
        print(pic.shape)
        np.save(f"/d/wangx/patches/skin/{0}.npy", pic.numpy())
        # i=i+1
        
if __name__ == "__main__":
    cut_pic()
# cut_pic()