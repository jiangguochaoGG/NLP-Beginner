from ast import arg
from distutils import core
import os, time, pickle, nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import wandb

from model import ESIM
from data_loader import SNLI
from vocab import Vocab

def train(model, optimizer, criterion, epoch, train_data_loader, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    total_data_num = len(train_data_loader.dataset)
    now_data_num = 0
    steps = 0
    for _, data in enumerate(train_data_loader):
        steps += 1
        now_data_num += 1
        optimizer.zero_grad()
        batch_sentence1, batch_sentence2, batch_label = data
        batch_input1 = np.zeros(shape=(len(batch_label), args.sentence_len))
        batch_input2 = np.zeros(shape=(len(batch_label), args.sentence_len))
        batch_label = batch_label.to(device)
        for i in range(len(batch_sentence1)):
            words = nltk.word_tokenize(batch_sentence1[i])
            for j in range(args.sentence_len):
                if j < len(words): batch_input1[i][j] = vocab.stoi[words[j]]
                else: batch_input1[i][j] = vocab.stoi['<pad>']
        for i in range(len(batch_sentence2)):
            words = nltk.word_tokenize(batch_sentence2[i])
            for j in range(args.sentence_len):
                if j < len(words): batch_input2[i][j] = vocab.stoi[words[j]]
                else: batch_input2[i][j] = vocab.stoi['<pad>']
        batch_input1 = torch.from_numpy(batch_input1).int().to(device)
        batch_input2 = torch.from_numpy(batch_input2).int().to(device)
        batch_len1, batch_len2 = [], []
        for i in range(len(batch_input1)):
            batch_len1.append(len(batch_input1[i]))
            batch_len2.append(len(batch_input2[i]))
        batch_len1 = torch.from_numpy(np.array(batch_len1)).int().to(device)
        batch_len2 = torch.from_numpy(np.array(batch_len2)).int().to(device)
        out = model(batch_input1, batch_len1, batch_input2, batch_len2)
        loss = criterion(out, batch_label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct += correct.item()
    print("Epoch %d:  Training average Loss: %f, Training accuracy: %f"
    %(epoch, total_loss/steps, total_correct*100/total_data_num))
    wandb.log(
        {'training_loss': total_loss/steps,
         'training_accuracy': total_correct*100/total_data_num,
         'epoch': epoch}
    )

def valid(model, criterion, epoch, val_data_loader, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    total_data_num = len(val_data_loader.dataset)
    steps = 0
    for _, data in enumerate(val_data_loader):
        steps += 1
        batch_sentence1, batch_sentence2, batch_label = data
        batch_input1 = np.zeros(shape=(len(batch_label), args.sentence_len))
        batch_input2 = np.zeros(shape=(len(batch_label), args.sentence_len))
        batch_label = batch_label.to(device)
        for i in range(len(batch_sentence1)):
            words = nltk.word_tokenize(batch_sentence1[i])
            for j in range(args.sentence_len):
                if j < len(words): batch_input1[i][j] = vocab.stoi[words[j]]
                else: batch_input1[i][j] = vocab.stoi['<pad>']
        for i in range(len(batch_sentence2)):
            words = nltk.word_tokenize(batch_sentence2[i])
            for j in range(args.sentence_len):
                if j < len(words): batch_input2[i][j] = vocab.stoi[words[j]]
                else: batch_input2[i][j] = vocab.stoi['<pad>']
        batch_input1 = torch.from_numpy(batch_input1).int().to(device)
        batch_input2 = torch.from_numpy(batch_input2).int().to(device)
        batch_len1, batch_len2 = [], []
        for i in range(len(batch_input1)):
            batch_len1.append(len(batch_input1[i]))
            batch_len2.append(len(batch_input2[i]))
        batch_len1 = torch.from_numpy(np.array(batch_len1)).int().to(device)
        batch_len2 = torch.from_numpy(np.array(batch_len2)).int().to(device)
        out = model(batch_input1, batch_len1, batch_input2, batch_len2)
        loss = criterion(out, batch_label)
        total_loss += loss.item()
        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct += correct.item()
    print("Epoch %d :  Validation average Loss: %f, Validation accuracy: %f"
      %(epoch, total_loss/steps, total_correct*100/total_data_num))
    wandb.log({'val_loss': total_loss/steps, 'val_accuracy': total_correct*100/total_data_num, 'epoch': epoch})
    global best_acc, best_epoch
    if best_acc < total_correct/total_data_num:
        best_acc = total_correct/total_data_num
        best_epoch = epoch
        path = './save/lr_{}'.format(args.lr)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model, path+'/best_model.pth')

def test(model, test_data_loader, device):
    total_correct = 0
    total_data_num = len(test_data_loader.dataset)
    steps = 0
    for _, data in enumerate(test_data_loader):
        steps += 1
        batch_sentence1, batch_sentence2, batch_label = data
        batch_input1 = np.zeros(shape=(len(batch_label), args.sentence_len))
        batch_input2 = np.zeros(shape=(len(batch_label), args.sentence_len))
        batch_label = batch_label.to(device)
        for i in range(len(batch_sentence1)):
            words = nltk.word_tokenize(batch_sentence1[i])
            for j in range(args.sentence_len):
                if j < len(words):
                    try:
                        batch_input1[i][j] = vocab.stoi[words[j]]
                    except:
                        batch_input1[i][j] = vocab.stoi['<unk>']
                else: batch_input1[i][j] = vocab.stoi['<pad>']
        for i in range(len(batch_sentence2)):
            words = nltk.word_tokenize(batch_sentence2[i])
            for j in range(args.sentence_len):
                if j < len(words):
                    try:
                        batch_input2[i][j] = vocab.stoi[words[j]]
                    except:
                        batch_input2[i][j] = vocab.stoi['<unk>']
                else: batch_input2[i][j] = vocab.stoi['<pad>']
        batch_input1 = torch.from_numpy(batch_input1).int().to(device)
        batch_input2 = torch.from_numpy(batch_input2).int().to(device)
        batch_len1, batch_len2 = [], []
        for i in range(len(batch_input1)):
            batch_len1.append(len(batch_input1[i]))
            batch_len2.append(len(batch_input2[i]))
        batch_len1 = torch.from_numpy(np.array(batch_len1)).int().to(device)
        batch_len2 = torch.from_numpy(np.array(batch_len2)).int().to(device)
        out = model(batch_input1, batch_len1, batch_input2, batch_len2)
        correct = (torch.max(out, dim=1)[1].view(batch_label.size()) == batch_label).sum()
        total_correct += correct.item()
    print("Test accuracy: %f" % (total_correct*100/total_data_num))
    wandb.log({'test_accuracy': total_correct*100/total_data_num})

if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('-vocab_path', type=str, default='./vocab.pkl', help='词表地址')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-embedding_dim', type=int, default=300, help='词嵌入向量维度')
    parser.add_argument('-hidden_size', type=int, default=128, help='LSTM隐藏层单元个数')
    parser.add_argument('-sentence_len', type=int, default=32, help='句子长度限制')
    args = parser.parse_args()
    wandb.init(project="zsgc-pj3", entity="qw8589177",
        name="ESIM{}D_lr:{:.4f}".format(args.embedding_dim, args.lr)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = SNLI(data_path='./data/snli_1.0/snli_1.0_train.txt')
    val_data = SNLI(data_path='./data/snli_1.0/snli_1.0_dev.txt')
    test_data = SNLI(data_path='./data/snli_1.0/snli_1.0_test.txt')
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = ESIM(num_embeddings=len(vocab.vocab), embedding_dim=args.embedding_dim, hidden_size=args.hidden_size,
                 embedding_choice='glove', vocab=vocab)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    starg_time = time.time()
    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.epoch+1):
        train(model, optimizer, criterion, epoch, train_data_loader, device)
        valid(model, criterion, epoch, val_data_loader, device)
        print('================================================================')
    print("Best val_acc:{}\t Best epoch:{}".format(best_acc, best_epoch))

    print('================================================================')
    print('Begin Test')
    model = torch.load('./save/lr_{}/best_model.pth'.format(args.lr))
    test(model, test_data_loader, device)
