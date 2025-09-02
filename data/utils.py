import pickle
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

    
def get_dataloader(args, data, weighted = False):

    train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
    dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
    test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)

    dataloader = {
        'train': train_dataloader,
        'dev': dev_dataloader,
        'test': test_dataloader
    }  
    return dataloader
