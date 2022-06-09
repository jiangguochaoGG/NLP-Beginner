from distutils.command.build import build
import pandas as pd
import numpy as np
import nltk
import pickle
import torch, torchtext

class Vocab():
    def __init__(self):
        self.vocab = set(["<pad>", "<unk>"])
        self.itos = {0: "<pad>", 1: "<unk>"}
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.vectors = []

    def build_vocab(self, texts):
        texts = texts.dropna()
        texts = texts.apply(lambda s: s.lower())
        for text in texts:
            words = nltk.word_tokenize(text)
            for word in words:
                if word in self.vocab:
                    continue
                else:
                    self.vocab.add(word)
                    self.stoi[word] = len(self.stoi)
                    self.itos[len(self.stoi)-1] = word
        self.build_vectors()

    def build_vectors(self, use_glove=True):
        if use_glove:
            glove = torchtext.vocab.GloVe(name='6B', dim=300, cache='../.vector_cache')
            num_embeddings = len(self.vocab)
            embedding_dim = glove.vectors.shape[1]
            self.vectors = np.zeros(shape=(num_embeddings, embedding_dim))
            for word in self.vocab:
                try:
                    self.vectors[self.stoi[word], :] = glove.vectors[glove.stoi[word], :]
                except:
                    if word == "<pad>": continue
                    self.vectors[self.stoi[word], :] = glove.vectors[glove.stoi["unk"], :]
            self.vectors = torch.from_numpy(self.vectors).float()

if __name__ == "__main__":
    save_path = "./vocab.pkl"
    train_data = pd.read_csv("./data/snli_1.0/snli_1.0_train.txt", sep='\t')
    val_data = pd.read_csv("./data/snli_1.0/snli_1.0_dev.txt", sep='\t')
    sentence1, sentence2 = train_data['sentence1'], train_data['sentence2']
    sentence1 = pd.concat([sentence1, val_data['sentence1']], axis=0)
    sentence2 = pd.concat([sentence2, val_data['sentence2']], axis=0)
    sentence1.columns = ['texts']
    sentence2.columns = ['texts']
    texts = pd.concat([sentence1, sentence2], axis=0)
    vocab = Vocab()
    vocab.build_vocab(texts)
    with open(save_path, "wb") as f:
        pickle.dump(vocab, f)