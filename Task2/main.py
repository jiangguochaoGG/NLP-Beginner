from operator import le
import os, time, pickle, nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torchtext
import wandb

from model import CNN, RNN, C_RNN
from data_loader import TextData
from vocab import Vocab

data_path = './data/'
batch_size = 64
learning_rate = 5e-4
embedding_choice = 'glove'
embedding_dim = 300
dropout = 0.5
out_channels = 128
label_num = 5
sentence_len = 32
epochs = 50
optim = "Adam"
model_type = "CNN"
rnn_type = "GRU"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="zsgc-pj2", entity="qw8589177", config={
    "learning_rate": learning_rate, "batch_size": batch_size,
    "embedding_choice": embedding_choice, "epochs": epochs,
    "optim": optim, "dropout": dropout, "model_type": model_type,
    "rnn_type": rnn_type
    }, name="After{}_lr:{:.4f}_embedding:{}_dropout".format(model_type, learning_rate, embedding_choice)
)

train_data_loader = TextData(data_path+"train_data.tsv", train=True)
val_data_loader = TextData(data_path+"val_data.tsv", train=True)
train_data_loader = DataLoader(train_data_loader, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data_loader, batch_size=batch_size, shuffle=True)
with open("./vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def train(model, epoch):
    model.train()
    total_loss=0.0
    total_correct=0.0
    total_data_num = len(train_data_loader.dataset)
    steps = 0
    for _, data in enumerate(train_data_loader):
        steps += 1
        optimizer.zero_grad()
        batch_text, batch_label = data
        batch_label = batch_label.to(device)
        batch_input = np.zeros(shape=(batch_label.size()[0], sentence_len))
        for i in range(len(batch_text)):
            words = nltk.word_tokenize(batch_text[i])
            for j in range(sentence_len):
                if j < len(words):
                    batch_input[i][j] = vocab.stoi[words[j]]
                else:
                    batch_input[i][j] = vocab.stoi['<pad>']
        batch_input = torch.from_numpy(batch_input).int().to(device)
        out = model(batch_input)
        loss = criterion(out, batch_label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct += correct.item()
    wandb.log({'training_loss': total_loss/steps, 'training_accuracy': total_correct*100/total_data_num, 'epoch': epoch+1})
    print("Epoch %d:  Training average Loss: %f, Training accuracy: %f, Total Time:%f"
    %(epoch+1, total_loss/steps, total_correct*100/total_data_num, time.time()-start_time))

def valid(model, epoch):
    model.eval()
    total_loss=0.0
    total_correct=0.0
    total_data_num = len(val_data_loader.dataset)
    steps = 0
    for _, data in enumerate(val_data_loader):
        steps+=1
        batch_text, batch_label = data
        batch_input = np.zeros(shape=(batch_label.size()[0], sentence_len))
        for i in range(len(batch_text)):
            words = nltk.word_tokenize(batch_text[i])
            for j in range(sentence_len):
                if j < len(words):
                    batch_input[i][j] = vocab.stoi[words[j]]
                else:
                    batch_input[i][j] = vocab.stoi['<pad>']
        batch_input = torch.from_numpy(batch_input).int().to(device)
        batch_label = batch_label.to(device)
        out = model(batch_input)
        loss = criterion(out, batch_label)
        total_loss += loss.item()
        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct += correct.item()
    wandb.log({'val_loss': total_loss/steps, 'val_accuracy': total_correct*100/total_data_num, 'epoch': epoch+1})
    print("Epoch %d :  Validation average Loss: %f, Validation accuracy: %f ,Total Time:%f"
      %(epoch+1, total_loss/steps, total_correct*100/total_data_num, time.time()-start_time))
    global best_accuracy, best_epoch
    if best_accuracy < total_correct/total_data_num :
        best_accuracy = total_correct/total_data_num
        best_epoch = epoch + 1

if model_type == "CNN":
    model=CNN(
        len(vocab.vocab), embedding_dim=embedding_dim, embedding_choice=embedding_choice,
        out_channels=out_channels, vocab=vocab, dropout=dropout, label_num=label_num
    )
elif model_type == "RNN":
    model = RNN(
        len(vocab.vocab), embedding_dim, embedding_choice, vocab, rnn_type, rnn_hidden_size=50,
        rnn_layers=2, dropout=0.5, label_num=label_num
    )
elif model_type == "C_RNN":
    model = C_RNN(
        len(vocab.vocab), embedding_dim, embedding_choice, out_channels, vocab, rnn_type,
        rnn_hidden_size=50, rnn_layers=2, dropout=0.5, label_num=label_num
    )
if optim == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
elif optim == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
start_time=time.time()
best_accuracy = 0.0
best_epoch = 0
for epoch in range(epochs):
    train(model, epoch)
    valid(model, epoch)
    print('============================================================================')
wandb.log({"score": best_accuracy, "best_epoch": best_epoch})
print("Best val_acc:{}\t Best epoch:{}".format(best_accuracy, best_epoch))