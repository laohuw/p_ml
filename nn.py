from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import torch
import requests
from pathlib import Path
import pickle
import gzip

DATA_PATH=Path("data")
FilePath=DATA_PATH/"mnist"
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

PATH.mkdir(parents=True, exist_ok=True)

if not (FilePath/FILENAME).exists():
    content=requests.get(URL+FILENAME).content
    (FilePath/FILENAME).open("w+b").write(content)
    



with gzip.open((FilePath/FILENAME).as_posix(),"rb") as f:
    ((x_train, y_train), (x_valid, y_valid),_)=pickle.load(f, encoding="latin-1")
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)   
(x_train, y_train, x_valid, y_valid)=map(torch.tensor, (x_train, y_train, x_valid, y_valid))

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super(Mnist_Logistic, self).__init__()
        self.lin=nn.Linear(784,10)
        
        
    def forward(self, xb):
        return self.lin(xb)
    
loss_func=F.cross_entropy
bs=64
lr=0.05
epochs=100
train_ds=TensorDataset(x_train, y_train)
train_dl=DataLoader(train_ds, batch_size=bs)

valid_ds=TensorDataset(x_valid, y_valid)
valid_dl=DataLoader(valid_ds, batch_size=bs*2)

def get_model():
    model=Mnist_Logistic()
    opt=optim.SGD(model.parameters(), lr=lr)
    return model, opt

model,opt=get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred=model(xb)
        loss=loss_func(pred, yb)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    with torch.no_grad():
        valid_loss=sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
    print(epoch, valid_loss/len(valid_dl))        
