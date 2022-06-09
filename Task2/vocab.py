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
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    texts = train_data['Phrase']
    vocab = Vocab()
    vocab.build_vocab(texts)
    with open(save_path, "wb") as f:
        pickle.dump(vocab, f)