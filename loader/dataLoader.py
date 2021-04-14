import numpy as np
from torch.utils.data import Dataset, DataLoader

class bikeDataNYC(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        # (4392, 2, 16, 8)
        self.data = np.load(args.data_path_NYC, allow_pickle=True)

        #(24, 300)
        self.hour_feature = np.load(args.data_path_hour_feature, allow_pickle=True)

        self.seq_len = args.seq_len

        if self.mode == 'train':
            self.data = self.data[:156*24]
        elif self.mode == 'val':
            self.data = self.data[156*24:-10*24]
        elif self.mode == 'test':
            self.data = self.data[-24*10:]

        self.data_valid_index = list(range(self.args.seq_len, len(self.data)))

    def __getitem__(self, index):
        item_index = self.data_valid_index[index]
        sample = []
        for i in range(self.seq_len):
            sample.append(self.data[item_index-i-1])
        sample.reverse()
        sample.append(self.data[item_index])

        sample.append(self.hour_feature[item_index % 24])
        # sample.append(item_index % 24)
        
        return sample

    def __len__(self):
        return len(self.data_valid_index)