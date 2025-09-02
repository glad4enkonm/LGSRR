import torch
from torch.utils.data import Dataset
__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, speaker_ids = None, multi_turn = False, other_hyper = None):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.size = len(self.text_data)
        self.speaker_ids = speaker_ids
        self.multi_turn = multi_turn

        self.other_hyper = other_hyper
        '''
        Note that the parameters in other_hypers cannot be the same as in the parent class
        '''

        if self.other_hyper is not None:
            for key in other_hyper.keys():
                setattr(self, key, other_hyper[key])  

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        
        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_data[index]),
        } 

        if self.other_hyper is not None:
            for key in self.other_hyper.keys():
                sample[key] = torch.tensor(getattr(self, key)[index])

        return sample
    