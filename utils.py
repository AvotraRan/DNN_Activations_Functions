import model.model as md
from  Data import dataloader

from train import train
from test import test
import torch
import torch.nn as nn
from config import args

criterion = nn.CrossEntropyLoss()



def my_function(a_f, num_epochs,use_pytorch = False):
  conv_model = md.DNN(28, 10, act_fc = a_f, use_pytorch_ac = use_pytorch)
  optimizer = torch.optim.SGD(conv_model.parameters(), lr=args.lr, momentum=args.momentum)
  train(model=conv_model, criterion=criterion, data_loader=dataloader.train_loader, optimizer=optimizer, num_epochs=num_epochs)
  test(conv_model, dataloader.test_loader,name_of_ac=a_f, use_pytorch=use_pytorch)

