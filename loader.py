import torch
from torch.utils.data import Dataset
import numpy as np

class BrainDataset(Dataset):
    def __init__(self,inputs_dir, outputs_dir):
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir

    def __len__(self):
        return len(self.inputs_dir)

    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(self.inputs_dir[idx]))
        mask = torch.from_numpy(np.load(self.outputs_dir[idx]))
        return image, mask