from torch.utils.data import Dataset
import pandas as pd

def load_text(data_path):
    data = pd.read_csv(data_path, sep='\t')
    texts = data['Phrase']
    texts = texts.apply(lambda s: s.lower())
    labels = data['Sentiment']
    return texts, labels

class TextData(Dataset):
    def __init__(self, data_path, train=True):
        if train:
            self.texts, self.labels = load_text(data_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        label = self.labels.iloc[index]
        return text, label