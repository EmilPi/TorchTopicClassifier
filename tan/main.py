# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
import json
import os
import time
from sys import argv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification

NGRAMS = 2
BATCH_SIZE = 64

EMBED_DIM = 32


def load_dataset():
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='./.data', ngrams=NGRAMS, vocab=None)
    ws2inds = train_dataset.get_vocab().itos

    return train_dataset, test_dataset, ws2inds


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.dropout = nn.Dropout(.7)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.25
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        dropped_embedded = self.dropout(embedded)
        return self.fc(dropped_embedded)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


text = ' '.join(argv[1:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Will run on %s' % device)

ws2inds = None
try:
    model = torch.load("topic_classifier_model", map_location=device)
    data = json.load(open('topic_classifier_vocabulary', 'r', encoding='utf-8'))
    ws2inds = data['ws2inds'].splitlines()
    VOCAB_SIZE = data['VOCAB_SIZE']
    EMBED_DIM = data['EMBED_DIM']
    NUM_CLASS = data['NUM_CLASS']

except Exception as e:
    print(e)
    train_dataset, test_dataset, ws2inds = load_dataset()

    VOCAB_SIZE = len(train_dataset.get_vocab())
    NUM_CLASS = len(train_dataset.get_labels())

    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
    try:
        model = torch.load("topic_classifier_model", map_location=device)
    except Exception as e:
        print(e)

if text != '':
    def prepare_text_for_tokenizing(text):
        text = text.lower().replace('.', ' .')
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text


    def text2inds(text):
        text = prepare_text_for_tokenizing(text)
        return [ws2inds.index(w) if w in ws2inds else ws2inds.index('<unk>') for w in text.lower().split()]


    def text2offsets(text_vecs):
        return [0] + [len(text_vecs)]

    
    while text != '':

        text_vecs_list = text2inds(text)
        text_offsets_list = text2offsets(text_vecs_list)

        text_offsets = torch.tensor(text_offsets_list[:-1]).cumsum(dim=0).to(device)
        text_vecs = torch.tensor(text_vecs_list).to(device)
        print(model(text_vecs, text_offsets))
        text = input('Enter a text to classify (ENTER to exit): ')
    exit()

else:
    train_dataset, test_dataset, ws2inds = load_dataset()

    N_EPOCHS = 5
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    torch.save(model, "topic_classifier_model")
    json.dump(dict(
        VOCAB_SIZE=VOCAB_SIZE,
        EMBED_DIM=EMBED_DIM,
        NUM_CLASS=NUM_CLASS,
        ws2inds='\n'.join(ws2inds),
    ),
        open('topic_classifier_vocabulary', 'w', encoding='utf-8'),
        ensure_ascii=False,
    )
