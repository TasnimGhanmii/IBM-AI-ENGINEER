# ----------------------------------------------------------
# Suppress warnings
# ----------------------------------------------------------
def warn(*args, **kwargs):
    pass
import warnings, math, pickle, torch, torch.nn as nn, numpy as np, pandas as pd
warnings.warn = warn
warnings.filterwarnings('ignore')

from tqdm import tqdm
from itertools import accumulate
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Positional-encoding module (unchanged)
# ----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ----------------------------------------------------------
# Text classifier
# ----------------------------------------------------------
class Net(nn.Module):
    def __init__(self, vocab_size, num_class, embedding_dim=100, nhead=5,
                 dim_feedforward=2048, num_layers=6, dropout=0.1,
                 activation="relu", classifier_dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim, dropout=dropout, vocab_size=vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embedding_dim, num_class)
        self.d_model = embedding_dim

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ----------------------------------------------------------
# Data pipelines
# ----------------------------------------------------------
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

def text_pipeline(x):          # string -> list[int]
    return vocab(tokenizer(x))

def label_pipeline(x):         # str-label -> int
    return int(x) - 1

def collate_batch(batch):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        text_list.append(torch.tensor(text_pipeline(_text), dtype=torch.int64))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

# ----------------------------------------------------------
# Build AG-News data-loaders
# ----------------------------------------------------------
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset  = to_map_style_dataset(test_iter)
num_train     = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train])

vocab = build_vocab_from_iterator(yield_tokens(AG_NEWS(split="train")),
                                  specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

BATCH_SIZE = 64
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)

# ----------------------------------------------------------
# Model, loss, optimiser
# ----------------------------------------------------------
vocab_size  = len(vocab)
num_class   = 4               # AG-News has 4 classes
model       = Net(vocab_size=vocab_size, num_class=num_class).to(device)
criterion   = nn.CrossEntropyLoss()
optimizer   = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for label, text in dataloader:
            pred = model(text)
            total_acc += (pred.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

# ----------------------------------------------------------
# Training loop
# ----------------------------------------------------------
EPOCHS = 10
cum_loss_list, acc_epoch = [], 0.0
best_acc = 0.0

for epoch in tqdm(range(1, EPOCHS + 1)):
    model.train()
    cum_loss = 0.0
    for label, text in train_dataloader:
        optimizer.zero_grad()
        label, text = label.to(device), text.to(device)
        pred = model(text)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        cum_loss += loss.item()

    cum_loss_list.append(cum_loss)
    val_acc = evaluate(valid_dataloader)
    print(f"Epoch {epoch:02d}  loss={cum_loss:.3f}  val-acc={val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "my_model.pth")

    scheduler.step()

# ----------------------------------------------------------
# Save & load metrics
# ----------------------------------------------------------
with open("loss.pkl", "wb") as f:
    pickle.dump(cum_loss_list, f)
with open("acc.pkl",  "wb") as f:
    pickle.dump([evaluate(valid_dataloader)], f)   # quick save

# ----------------------------------------------------------
# Final test evaluation
# ----------------------------------------------------------
test_acc = evaluate(test_dataloader)
print(f"Final test accuracy: {test_acc:.3f}")