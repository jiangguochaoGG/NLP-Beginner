import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones([batch_size, max_length], dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0
    return mask

def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor*mask+values_to_add

class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1,
        bias=True, dropout=0.0, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.encoder = rnn_type(input_size, hidden_size, num_layers=num_layers,
                                bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_lengths, sorting_index = sequences_lengths.sort(0, descending=False)
        sorted_batch = sequences_batch.index_select(0, sorting_index)
        idx_range = torch.arange(0, len(sequences_batch)).type_as(sequences_lengths)
        _, reverse_mapping = sorting_index.sort(0, descending=False)
        restoration_idx = idx_range.index_select(0, reverse_mapping)

        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths.cpu(), batch_first=True)

        outputs, _ = self.encoder(packed_batch)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        restoration_idx = restoration_idx.to(sequences_batch.device)
        reordered_outputs = outputs.index_select(0, restoration_idx)
        return reordered_outputs

class SoftmaxAttention(nn.Module):
    def masked_softmax(self, batch, mask):
        batch_shape = batch.size()
        reshaped_batch = batch.view(-1, batch_shape[-1])
        while mask.dim() < batch.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(batch).contiguous().float()
        reshaped_mask = mask.view(-1, mask.size()[-1])
        result = F.softmax(reshaped_mask*reshaped_batch, dim=-1)
        return result.view(batch_shape)

    def weighted_sum(self, tensor, weights, mask):
        weighted_sum = weights.bmm(tensor)
        while mask.dim() < weighted_sum.dim():
            mask = mask.unsqueeze(1)
        mask = mask.transpose(-1, -2)
        mask = mask.expand_as(weighted_sum).contiguous().float()
        return weighted_sum*mask

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())

        prem_hyp_attn = self.masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = self.masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)

        attended_premises = self.weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = self.weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses

class ESIM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, embedding_choice,
        vocab, dropout=0.5, label_nums=5):
        super(ESIM, self).__init__()

        if embedding_choice == 'glove':
            self.embedding = nn.Embedding(
                num_embeddings, embedding_dim
            ).from_pretrained(vocab.vectors, freeze=True)
        elif embedding_choice == 'rand':
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.rnn_encoder = RNNEncoder(rnn_type=nn.LSTM, input_size=embedding_dim,
                                      hidden_size=hidden_size, bidirectional=True)
        self.attention = SoftmaxAttention()
        self.projection = nn.Sequential(
            nn.Linear(8*hidden_size, hidden_size)
        )
        self.composition = RNNEncoder(nn.LSTM, hidden_size, hidden_size, bidirectional=True)
        self.classification = nn.Sequential(
            self.dropout,
            nn.Linear(8*hidden_size, hidden_size),
            nn.Tanh(),
            self.dropout,
            nn.Linear(hidden_size, label_nums)
        )

    def forward(self, premises, premises_lengths, hypotheses, hypotheses_lengths):
        embedded_premises = self.embedding(premises)
        embedded_hypotheses = self.embedding(hypotheses)
        premises_mask = get_mask(premises, premises_lengths).cuda()
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).cuda()

        encoded_premises = self.rnn_encoder(embedded_premises, premises_lengths)
        encoded_hypotheses = self.rnn_encoder(embedded_hypotheses, hypotheses_lengths)

        attended_premises, attended_hypotheses = self.attention(encoded_premises, premises_mask,
                                                                encoded_hypotheses, hypotheses_mask)
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)
        projected_premises = self.projection(enhanced_premises)
        projected_hypotheses = self.projection(enhanced_hypotheses)

        v_ai = self.composition(projected_premises, premises_lengths)
        v_bj = self.composition(projected_hypotheses, hypotheses_lengths)
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(premises_mask, dim=1,
                                                                                                  keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(hypotheses_mask,
                                                                                                    dim=1, keepdim=True)
        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        out = self.classification(v)
        return out