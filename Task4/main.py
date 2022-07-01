import torch
import torch.optim as optim
from tqdm import tqdm
from model import NER_Model
from data_process import load_data, get_chunks
import wandb
from torchtext.vocab import Vectors

# wandb.init(
#     project="zsgc-pj4", entity="qw8589177",
#     name="3:CRF:false"
# )

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "data"
BEST_MODEL = "best_model.ckpt"
BATCH_SIZE = 64 # differ from 16
LOWER_CASE = False
EPOCHS = 50


# WORD_VECTORS = None # random
WORD_VECTORS = Vectors('glove.6B.100d.txt', '../.vector_cache')
WORD_EMBEDDING_SIZE = 100
CHAR_VECTORS = None
CHAR_EMBEDDING_SIZE = 30
FREEZE_EMBEDDING = False

LEARNING_RATE = 0.015
DECAY_RATE = 0.05
MOMENTUM = 0.9
CLIP = 5
PATIENCE = 5

HIDDEN_SIZE = 400
LSTM_LAYER_NUM = 1
DROPOUT_RATE = 0.5
USE_CHAR = True
N_FILTERS = 30
KERNEL_STEP = 3
USE_CRF = True

def train(train_loader, dev_loader, optimizer):
    best_dev_f1 = -1
    best_dev_loss = 9999
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        train_loader.init_epoch()
        for i, batch in enumerate(tqdm(train_loader)):
            words, lens = batch.word
            labels = batch.label
            model.zero_grad()
            words = words.to(DEVICE)
            lens = lens.to(DEVICE)
            labels = labels.to(DEVICE)
            loss = model(words, batch.char, lens, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %.2f" % (epoch, total_loss/(i)))
        # wandb.log(
        #     {'epoch': epoch, 'train loss': total_loss/i}
        # )
        lr = LEARNING_RATE / (1 + DECAY_RATE * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        dev_f1,dev_p,dev_r,dev_loss = eval(dev_loader, "Dev", epoch)

        # wandb.log(
        #     {'epoch': epoch, 'val f1': dev_f1, 'val precision': dev_p,
        #     'val recall': dev_r, 'val loss': dev_loss}
        # )

        # if dev_f1 < best_dev_f1:
        #     patience_counter += 1
        # else:
        #     best_dev_f1 = dev_f1
        #     patience_counter = 0
        #     torch.save(model.state_dict(), BEST_MODEL)

        if dev_loss > best_dev_loss:
            patience_counter += 1
        else:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL)

        # if patience_counter >= PATIENCE:
        #     tqdm.write("Early stop!")
        #     break


def eval(data_loader, name, epoch=None, best_model=None):
    if best_model:
        model.load_state_dict(torch.load(best_model))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        result = [0,0,0]
        for i, batch in enumerate(data_loader):
            words, lens = batch.word
            labels = batch.label
            predicted_seq, _ = model(words, batch.char, lens)
            loss = model(words, batch.char, lens, labels)
            total_loss += loss.item()

            orig_text = [e.word for e in data_loader.dataset.examples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            for text, ground_truth_id, predicted_id, len_ in zip(orig_text, labels.cpu().numpy(),
                                                                 predicted_seq.cpu().numpy(),
                                                                 lens.cpu().numpy()):
                lab_chunks = set(get_chunks(ground_truth_id[:len_], LABEL.vocab.stoi,flag=False))
                lab_pred_chunks = set(get_chunks(predicted_id[:len_], LABEL.vocab.stoi,flag=False))

                for chunk in list(lab_chunks):
                    if chunk in lab_pred_chunks:
                        result[0] += 1
                    result[2] += 1
                for _ in list(lab_pred_chunks):
                    result[1] += 1

        loss = total_loss/i

        precision = result[0] / result[1] if result[1] != 0 else 0
        recall = result[0] / result[2] if result[2] != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        print(name+' F1: %.2f, Recall: %.2f, Precision: %.2f' % (f1,recall,precision))
    return f1,precision,recall,loss




if __name__ == "__main__":
    train_iter, dev_iter, test_iter, WORD, CHAR, LABEL = load_data(WORD_EMBEDDING_SIZE,
                                                                   WORD_VECTORS,
                                                                   CHAR_EMBEDDING_SIZE,
                                                                   CHAR_VECTORS,
                                                                   BATCH_SIZE,
                                                                   DEVICE,
                                                                   DATA_PATH)
    model = NER_Model(WORD.vocab.vectors,
                      CHAR.vocab.vectors,
                      len(LABEL.vocab.stoi),
                      HIDDEN_SIZE,
                      DROPOUT_RATE,
                      KERNEL_STEP,
                      N_FILTERS,
                      USE_CHAR,
                      FREEZE_EMBEDDING,
                      USE_CRF).to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    train(train_iter, dev_iter, optimizer)
    test_f1,test_p,test_r,test_loss = eval(test_iter, "Test", best_model=BEST_MODEL)
    # wandb.log(
    #     {'test f1': test_f1, 'test precision': test_p, 'test recall': test_r, 'test loss': test_loss}
    # )
