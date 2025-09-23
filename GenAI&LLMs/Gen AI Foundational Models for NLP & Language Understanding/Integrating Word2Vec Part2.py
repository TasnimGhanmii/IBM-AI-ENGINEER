
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

# =====================
# Device setup
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Visualization
# =====================
def plot_embeddings(word_embeddings, vocab, max_points=200):
    tsne = TSNE(n_components=2, random_state=0)
    word_embeddings_2d = tsne.fit_transform(word_embeddings[:max_points])

    plt.figure(figsize=(15, 15))
    for i, word in enumerate(vocab.get_itos()[:max_points]):
        plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
        plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))

    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title("Word Embeddings visualized with t-SNE")
    plt.show()


# =====================
# Cosine Similarity
# =====================
def find_similar_words(word, word_embeddings, vocab, top_k=5):
    stoi = vocab.get_stoi()
    itos = vocab.get_itos()

    if word not in stoi:
        print("Word not found in vocab.")
        return []

    target_idx = stoi[word]
    target_vector = word_embeddings[target_idx]

    similarities = {}
    for i in range(len(word_embeddings)):
        if i != target_idx:
            sim = torch.dot(torch.tensor(target_vector), torch.tensor(word_embeddings[i])) / (
                torch.norm(torch.tensor(target_vector)) * torch.norm(torch.tensor(word_embeddings[i]))
            )
            similarities[itos[i]] = sim.item()

    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_k]]


# =====================
# Training Function
# =====================
def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    epoch_losses = []

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            if isinstance(model, CBOW):
                target, context, offsets = batch
                output = model(context, offsets)
            else:  # Skip-gram
                target, context = batch
                output = model(context)

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += loss.item()

        epoch_losses.append(running_loss / len(dataloader))

    return model, epoch_losses


# =====================
# Data
# =====================
toy_data = """I wish I was little bit taller I wish I was a baller ..."""  # Truncated for brevity
tokenizer = get_tokenizer("basic_english")
tokenized_toy_data = tokenizer(toy_data)

def yield_tokens(data):
    yield tokenizer(data)

vocab = build_vocab_from_iterator([tokenized_toy_data], specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

text_pipeline = lambda tokens: [vocab[token] for token in tokens]

# =====================
# CBOW Setup
# =====================
CONTEXT_SIZE = 2
cobow_data = []

for i in range(CONTEXT_SIZE, len(tokenized_toy_data) - CONTEXT_SIZE):
    context = [tokenized_toy_data[i - j - 1] for j in range(CONTEXT_SIZE)][::-1] + \
              [tokenized_toy_data[i + j + 1] for j in range(CONTEXT_SIZE)]
    target = tokenized_toy_data[i]
    cobow_data.append((context, target))

def collate_cbow(batch):
    target_list, context_list, offsets = [], [], [0]
    for context, target in batch:
        target_list.append(vocab[target])
        processed = torch.tensor(text_pipeline(context), dtype=torch.int64)
        context_list.append(processed)
        offsets.append(processed.size(0))
    target_list = torch.tensor(target_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    context_list = torch.cat(context_list)
    return target_list.to(device), context_list.to(device), offsets.to(device)

cbow_loader = DataLoader(cobow_data, batch_size=64, shuffle=True, collate_fn=collate_cbow)

# =====================
# CBOW Model
# =====================
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.linear1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc = nn.Linear(embed_dim // 2, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        out = self.embedding(text, offsets)
        out = torch.relu(self.linear1(out))
        return self.fc(out)

vocab_size = len(vocab)
embed_dim = 24
model_cbow = CBOW(vocab_size, embed_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_cbow.parameters(), lr=5)

model_cbow, cbow_losses = train_model(model_cbow, cbow_loader, criterion, optimizer, num_epochs=100)
plt.plot(cbow_losses)
plt.title("CBOW Training Loss")
plt.show()

cbow_embeddings = model_cbow.embedding.weight.detach().cpu().numpy()
plot_embeddings(cbow_embeddings, vocab)

# =====================
# Skip-gram Setup
# =====================
skip_data = []
for i in range(CONTEXT_SIZE, len(tokenized_toy_data) - CONTEXT_SIZE):
    target = tokenized_toy_data[i]
    context = [tokenized_toy_data[i - j - 1] for j in range(CONTEXT_SIZE)] + \
              [tokenized_toy_data[i + j + 1] for j in range(CONTEXT_SIZE)]
    skip_data.extend([(target, ctx) for ctx in context])

def collate_skipgram(batch):
    targets, contexts = [], []
    for target, context in batch:
        targets.append(vocab[target])
        contexts.append(vocab[context])
    return torch.tensor(targets, dtype=torch.int64).to(device), \
           torch.tensor(contexts, dtype=torch.int64).to(device)

skip_loader = DataLoader(skip_data, batch_size=64, shuffle=True, collate_fn=collate_skipgram)

# =====================
# Skip-gram Model
# =====================
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = torch.relu(out)
        return self.fc(out)

model_sg = SkipGram(vocab_size, embed_dim).to(device)
optimizer = optim.SGD(model_sg.parameters(), lr=5)

model_sg, sg_losses = train_model(model_sg, skip_loader, criterion, optimizer, num_epochs=100)
plt.plot(sg_losses)
plt.title("Skip-gram Training Loss")
plt.show()

sg_embeddings = model_sg.embeddings.weight.detach().cpu().numpy()
plot_embeddings(sg_embeddings, vocab)
