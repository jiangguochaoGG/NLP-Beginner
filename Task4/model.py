import torch
import math
from torch import nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, num_filters, kernel_sizes, padding):
        super(CharCNN, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, kernel_sizes, padding=padding)
    def forward(self, x):
        x = self.conv(x).squeeze(-1)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(-1)
        return x_max

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_size, hidden_size // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, lens):
        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]
        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        recover_index = index.argsort()
        output = output[recover_index]
        return output

class SoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim):
        super(SoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.linear = torch.nn.Linear(input_dim, label_size)

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.linear(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, lens, label_ids=None):
        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)
        predict_mask = (torch.arange(inputs.size(1)).expand(len(lens), inputs.size(1))).to(lens.device) < lens.unsqueeze(1)
        if label_ids is not None:
            # cross entropy loss
            p = torch.nn.functional.softmax(logits, -1)
            one_hot_labels = torch.eye(self.label_size)[label_ids].type_as(p)
            losses = -torch.log(torch.sum(one_hot_labels * p, -1))
            masked_losses = torch.masked_select(losses, predict_mask)
            return masked_losses.sum()
        else:
            return torch.argmax(logits, -1), p

def log_sum_exp(input, dim=0):
    m, _ = torch.max(input, dim)
    m_exp = m.unsqueeze(-1).expand_as(input)
    return m + torch.log(torch.sum(torch.exp(input - m_exp), dim))

class CRF(nn.Module):
    def __init__(self, label_size):
        super(CRF, self).__init__()

        self.label_size = label_size
        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.transition.data, -0.1, 0.1)
        self.transition.data[:, self.end] = -1000.0
        self.transition.data[self.start, :] = -1000.0

    def pad_logits(self, logits):
        batch_size, seq_len, label_num = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -1000.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, predict_mask):

        batch_size, seq_len = labels.size()

        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        labels_ext[:, -1] = self.end
        pad = predict_mask.new_ones([batch_size, 1], requires_grad=False)
        pad_stop = labels.new_full([batch_size, 1], self.end, requires_grad=False)
        mask = torch.cat([pad, predict_mask, pad], dim=-1).long()
        labels = (1 - mask) * pad_stop + mask * labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = torch.cat([pad, predict_mask], dim=-1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, predict_mask):

        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        scores = scores * predict_mask.float()
        return scores

    def calc_gold_score(self, logits, labels, predict_mask):

        unary_score = self.calc_unary_score(logits, labels, predict_mask).sum(
            1).squeeze(-1)

        binary_score = self.calc_binary_score(labels, predict_mask).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, predict_mask):

        batch_size, seq_len, feat_dim = logits.size()

        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0

        predict_mask_ = predict_mask.clone()

        logits_t = logits.transpose(1, 0)
        predict_mask_ = predict_mask_.transpose(1, 0)
        for word_mask_, logit in zip(predict_mask_, logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transition.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transition.size())
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = word_mask_.float().unsqueeze(-1).expand_as(alpha)  # (batch_size, num_labels+2)
            alpha = mask * alpha_nxt + (1 - mask) * alpha

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, predict_mask):

        batch_size, seq_len, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        predict_mask_ = predict_mask.clone()
        predict_mask_ = predict_mask_.transpose(1, 0)  # (max_seq, batch_size)
        logits_t = logits.transpose(1, 0)
        pointers = []
        for ix, logit in enumerate(logits_t):
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = predict_mask_[ix].float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (predict_mask_[ix:].sum(0) == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths
class CRFDecoder(nn.Module):
    def __init__(self, label_size, input_dim):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(in_features=input_dim,out_features=label_size)
        self.crf = CRF(label_size + 2)
        self.label_size = label_size

        self.init_weights()

    def init_weights(self):
        bias = math.sqrt(6 / (self.linear.weight.size(0) + self.linear.weight.size(1)))
        nn.init.uniform_(self.linear.weight, -bias, bias)

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.linear(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, lens, labels=None):

        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)
        logits = self.crf.pad_logits(logits)
        predict_mask = (torch.arange(inputs.size(1)).expand(len(lens), inputs.size(1))).to(lens.device) < lens.unsqueeze(1)
        if labels is None:
            _, preds = self.crf.viterbi_decode(logits, predict_mask)
            return preds, p
        return self.neg_log_likehood(logits, predict_mask, labels)

    def neg_log_likehood(self, logits, predict_mask, labels):
        norm_score = self.crf.calc_norm_score(logits, predict_mask)
        gold_score = self.crf.calc_gold_score(logits, labels, predict_mask)
        loglik = gold_score - norm_score
        return -loglik.sum()

class NER_Model(nn.Module):
    def __init__(self, word_embedding, char_embedding,num_labels, hidden_size, dropout_rate=0.5, kernel_step=3, char_out_size=100, use_char=False,
                 freeze=False, use_crf=True):
        super(NER_Model, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze)
        self.word_embedding_size = word_embedding.size(-1)
        self.use_char = use_char
        if use_char:
            self.char_embedding = nn.Embedding.from_pretrained(char_embedding, freeze)
            self.char_embedding_size = char_embedding.size(-1)
            self.charcnn = CharCNN(char_out_size, (kernel_step, self.char_embedding_size), (2, 0))
            self.bilstm = BiLSTM(char_out_size + self.word_embedding_size, hidden_size)
        else:
            self.bilstm = BiLSTM(self.word_embedding_size, hidden_size)

        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.rnn_in_dropout = nn.Dropout(dropout_rate)

        if use_crf:
            self.decoder = CRFDecoder(num_labels, hidden_size)
        else:
            self.decoder = SoftmaxDecoder(num_labels, hidden_size)


    def forward(self, word_ids, char_ids, lens, label_ids=None):
        word_embedding = self.word_embedding(word_ids)
        if self.use_char:
            char_embedding = self.char_embedding(char_ids).reshape(-1, char_ids.size(-1), self.char_embedding_size).unsqueeze(1)
            char_embedding = self.embedding_dropout(char_embedding)
            char_embedding = self.charcnn(char_embedding)
            char_embedding = char_embedding.reshape(char_ids.size(0), char_ids.size(1), -1)
            embedding = torch.cat([word_embedding, char_embedding], -1)
        else:
            embedding = word_embedding
        x = self.rnn_in_dropout(embedding)
        x = self.bilstm(x, lens.cpu())  # (batch_size, max_seq_len, hidden_size)
        x = self.out_dropout(x)
        return self.decoder(x, lens, label_ids)
