import numpy as np
from torchtext.legacy import data
from torchtext.legacy.data import Iterator, BucketIterator
import re
import os
import torch
import math
def read_data(data_path = r'C:\Users\laungee\PycharmProjects\pythonProject\ner\data\dev.txt'):
    lines = []
    words = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
        data = data.split('\n')

    for item in data:
        if item == '-DOCSTART- -X- -X- O':
            continue
        item = item.strip()
        if len(item) == 0:
            if len(words) == 0:
                continue
            lines.append([words, [list(word) for word in words], labels])
            words = []
            labels = []
            continue
        items = item.split(' ')
        word = items[0]
        label = items[-1]
        words.append(word)
        labels.append(label)
        pass

    return lines # (sentence,chars,labels)
def unk_init(x):
    """
    char random embedding according to the paper:
    """
    dim = x.size(-1)
    bias = math.sqrt(3.0 / dim)
    x.uniform_(-bias, bias)
    return x

class ConllDataset(data.Dataset):
    def __init__(self, word_field, char_field, label_field, data_path):
        self.fields = [("word", word_field), ("char", char_field), ("label", label_field)]
        datas = read_data(data_path)
        examples = []
        for word, char, label in datas:
            examples.append(data.Example.fromlist([word, char, label], self.fields))
        self.examples = examples
        self.sentence_num = len(self.examples)
        self.words_num = sum([len(example.word) for example in self.examples ])
        super(ConllDataset, self).__init__(self.examples,self.fields)

def get_chunk_type(token_id, id_to_label):
    label_name = id_to_label[token_id]
    tag_class,tag_type = label_name.split('-')
    return tag_class, tag_type

def get_chunks(label, label_id, bioes=True,flag = True):
    if flag == False:
        a = 1
    if flag:
        label = [label_id[i] for i in label]

    default = label_id["O"]

    idx_to_tag = {idx: tag for tag, idx in label_id.items()}

    chunks = []

    chunk_class, chunk_type, chunk_start = None, None, None
    for i, token in enumerate(label):
        if token == default and (chunk_class in (["E", "S"] if bioes else ["B", "I"])):
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_class, chunk_type, chunk_start = "O", None, None

        if token != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(token, idx_to_tag)
            if chunk_type is None:
                chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
            else:
                if bioes:
                    if chunk_class in ["E", "S"]:
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        if tok_chunk_class in ["B", "S"]:
                            chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
                        else:
                            chunk_class, chunk_type, chunk_start = None, None, None
                    elif tok_chunk_type == chunk_type and chunk_class in ["B", "I"]:
                        chunk_class = tok_chunk_class
                    else:
                        chunk_class, chunk_type = None, None
                else:
                    if tok_chunk_class == "B":
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
                    else:
                        chunk_class, chunk_type = None, None

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(label))
        chunks.append(chunk)

    return chunks

def get_vectors(embeddings, vocab, pretrain_embed_vocab):
    oov = 0
    for i, word in enumerate(vocab.itos):
        index = pretrain_embed_vocab.stoi.get(word, None)  # digit or None
        if index is None:
            if word.lower() in pretrain_embed_vocab.stoi:
                index = pretrain_embed_vocab.stoi[word.lower()]
        if index:
            embeddings[i] = pretrain_embed_vocab.vectors[index]
        else:
            oov += 1
    return embeddings

def load_data(word_embedding_size, word_vectors, char_embedding_size, char_vectors, batch_size=32, device="cpu",
               data_path='../code'):
    zero_char_in_word = lambda ex: [re.sub('\d', '0', w) for w in ex]
    zero_char = lambda w: [re.sub('\d', '0', c) for c in w]

    WORD_TEXT = data.Field(lower=True, batch_first=True, include_lengths=True, preprocessing=zero_char_in_word)
    CHAR_NESTING = data.Field(tokenize=list, preprocessing=zero_char)  # process a word in char list
    CHAR_TEXT = data.NestedField(CHAR_NESTING)
    LABEL = data.Field(unk_token=None, pad_token="O", batch_first=True)

    train_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "train.txt"))
    dev_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "dev.txt"))
    test_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "test.txt"))

    LABEL.build_vocab(train_data.label)
    WORD_TEXT.build_vocab(train_data.word, max_size=50000, min_freq=1)
    CHAR_TEXT.build_vocab(train_data.char, max_size=50000, min_freq=1)

    vectors_to_use = unk_init(torch.zeros((len(WORD_TEXT.vocab), word_embedding_size)))
    if word_vectors is not None:
        vectors_to_use = get_vectors(vectors_to_use, WORD_TEXT.vocab, word_vectors)
    WORD_TEXT.vocab.vectors = vectors_to_use

    vectors_to_use = unk_init(torch.zeros((len(CHAR_TEXT.vocab), char_embedding_size)))
    if char_vectors is not None:
        vectors_to_use = get_vectors(vectors_to_use, CHAR_TEXT.vocab, char_vectors)
    CHAR_TEXT.vocab.vectors = vectors_to_use

    # data loader
    train_loader = BucketIterator(train_data, batch_size=batch_size, device=device, sort_key=lambda x: len(x.word),
                                sort_within_batch=True, repeat=False, shuffle=True)
    dev_loader = Iterator(dev_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                        repeat=False, shuffle=False)
    test_loader = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_loader,dev_loader,test_loader,WORD_TEXT, CHAR_TEXT, LABEL

if __name__ == "__main__":
    batch_size = 16
    word_vectors = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_vectors = None
    word_embedding_size = 100
    char_vectors = None
    char_embedding_size = 30
    FREEZE_EMBEDDING = False
    train_loader,_,_,_,_,_ = load_data(word_embedding_size,word_vectors,char_embedding_size,char_vectors,batch_size)
    a = 1