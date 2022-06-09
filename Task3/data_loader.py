import numpy as np
from torch.utils.data import Dataset
import pandas as pd

class SNLI(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, sep='\t')
        data = data.dropna() # 去除nan数据
        data = data[data.gold_label != '-'] # 去除gold_label不显著的数据
        self.sentence1, self.sentence2 = [], []
        self.labels = []
        label2class = {
            'entailment': 0, 'neutral': 1, 'contradiction': 2
        }
        for i in range(len(data['gold_label'])):
            label = data['gold_label'].iloc[i]
            if label in label2class:
                self.labels.append(label2class[label])
                self.sentence1.append(data['sentence1'].iloc[i].lower())
                self.sentence2.append(data['sentence2'].iloc[i].lower())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sentence1 = self.sentence1[index]
        sentence2 = self.sentence2[index]
        label = self.labels[index]
        return sentence1, sentence2, label