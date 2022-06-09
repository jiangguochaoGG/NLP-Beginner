import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embedding_choice, out_channels, vocab, dropout=False, label_num=5):
        super(CNN, self).__init__()
        if embedding_choice == 'glove':
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim
            ).from_pretrained(vocab.vectors, freeze=True)
        elif embedding_choice == 'rand':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                               kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                               kernel_size=5, padding=2)

        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                               kernel_size=7, padding=3)

        self.dropout = nn.Dropout(dropout) if dropout else nn.Sequential()
        self.fc1 = nn.Linear(out_channels*3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, label_num)

    def forward(self,x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        out = self.fc3(x)
        return out

class RNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embedding_choice,
        vocab, rnn_type, rnn_hidden_size=50, rnn_layers=2, dropout=False, label_num=5):
        super(RNN, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        if embedding_choice == 'glove':
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim
            ).from_pretrained(vocab.vectors, freeze=True)
        elif embedding_choice == 'rand':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn_type = rnn_type
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers = rnn_layers,
                bidirectional = True, batch_first = True
            )
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers = rnn_layers,
                bidirectional = True, batch_first = True
            )

        self.dropout = nn.Dropout(dropout) if dropout else nn.Sequential()
        self.fc = nn.Linear(2*rnn_hidden_size, label_num)


    def forward(self, x):
        h = torch.zeros(self.rnn_layers*2, x.size(0), self.rnn_hidden_size).cuda()
        c = torch.zeros(self.rnn_layers*2, x.size(0), self.rnn_hidden_size).cuda()
        x = self.embedding(x)
        if self.rnn_type == "LSTM":
            res, _ = self.rnn(x, (h, c))
        elif self.rnn_type == "GRU":
            res, _ = self.rnn(x, h)
        res = self.dropout(res)
        res = self.fc(res[:, -1, :])
        return res

class C_RNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embedding_choice, out_channels,
        vocab, rnn_type, rnn_hidden_size=50, rnn_layers=2, dropout=False, label_num=5):
        super(C_RNN, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        if embedding_choice == 'glove':
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim
            ).from_pretrained(vocab.vectors, freeze=True)
        elif embedding_choice == 'rand':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                               kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                               kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers = rnn_layers,
                bidirectional = True, batch_first = True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers = rnn_layers,
                bidirectional = True, batch_first = True
            )

        self.dropout = nn.Dropout(dropout) if dropout else nn.Sequential()
        # self.fc = nn.Linear(3*out_channels+2*rnn_hidden_size, label_num)
        self.fc1 = nn.Linear(3*out_channels+2*rnn_hidden_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, label_num)


    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(self.rnn_layers*2, x.size(0), self.rnn_hidden_size).cuda()
        c = torch.zeros(self.rnn_layers*2, x.size(0), self.rnn_hidden_size).cuda()
        if self.rnn_type == "LSTM":
            rnn_feature, _ = self.rnn(x, (h, c))
        elif self.rnn_type == "GRU":
            rnn_feature, _ = self.rnn(x, h)

        cnn_x = torch.transpose(x, 1, 2)
        x1 = F.relu(self.bn(self.conv1(cnn_x)))
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.relu(self.bn(self.conv2(cnn_x)))
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.relu(self.bn(self.conv3(cnn_x)))
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        cnn_feature = torch.cat((x1, x2, x3), dim=1)

        feature = torch.cat((cnn_feature, rnn_feature[:, -1, :]), dim=1)
        feature = self.dropout(feature)
        feature = self.dropout(F.relu(self.fc1(feature)))
        feature = self.dropout(F.relu(self.fc2(feature)))
        res = self.fc3(feature)
        return res