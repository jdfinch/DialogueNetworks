
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

from conversation import *

conversations = Conversations.load('conversation_graphs.pckl')

features = 300
classes = len(conversations.concepts)

class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.W = torch.nn.Parameter(torch.rand(self.decoder_dim, self.encoder_dim))

    def forward(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor, # [seq_length, encoder_dim]
        ):
        weights = query @ self.W @ values.T  # [seq_length]
        weights = weights / np.sqrt(self.decoder_dim)  # [seq_length]
        weights = torch.nn.functional.softmax(weights, dim=0)
        return weights @ values  # [encoder_dim]

class GraphInferencer(torch.nn.Module):

    def __init__(self):
        super(GraphInferencer, self).__init__()
        self.conv = GATConv(features, features)
        self.att = Attention(features, features * 2)
        self.lin = torch.nn.Linear(features, classes)
        self.test1 = torch.nn.Linear(features * 2, features * 2)
        self.test2 = torch.nn.Linear(features * 2, features * 2)
        self.test3 = torch.nn.Linear(features * 2, classes)

    def forward(self, features, edges, query):
        # x = self.conv(features, edges)
        # x = F.elu(x)
        # x = self.att(query, x)
        # x = F.relu(x)
        # x = self.lin(x)
        x = self.test1(query)
        x = F.relu(x)
        x = self.test2(x)
        x = F.relu(x)
        x = self.test3(x)
        x = F.relu(x)
        # x = torch.nn.functional.softmax(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphInferencer().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

def train(features, edges, queries, targets):
    print('train MRR:', evaluate(features, edges, queries, targets))
    batch = 1000
    epochs = 1000
    model.train()
    running_loss = 0.0
    for e in range(epochs):
        print('Epoch', e, '-----------')
        for i in range(len(features)):
            optimizer.zero_grad()
            feature = features[i]
            edge = edges[i]
            target = targets[i]
            query = queries[i]
            output = model(feature, edge, query)
            loss = criterion(output.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        print('%5d loss: %.3f' % (e, running_loss / batch))
        running_loss = 0.0
        print('train MRR:', evaluate(features, edges, queries, targets))


def evaluate(features, edges, queries, targets):
    model.eval()
    with torch.no_grad():
        correct = 0
        mrr_num = 0
        mrr_denom = 0
        for i in range(len(features)):
            feature, edge, query = features[i], edges[i], queries[i]
            output = model(feature, edge, query)
            sortout, indices = torch.sort(output, descending=True)
            predicted = indices[0].item()
            cls = targets[i].item()
            if predicted == cls:
                correct += 1
            rank = ((indices == cls).nonzero()).item()
            mrr_num += 1
            mrr_denom += rank
            model.train()
        return mrr_num / mrr_denom

if __name__ == '__main__':
    print('Features to gpu...')
    features = [x.to(device) for x in conversations.features_tensors]
    print('Edges to gpu...')
    edges = [x.to(device).transpose(0, 1) for x in conversations.edges_tensors]
    print('Queries to gpu...')
    queries = [torch.cat((x[0], x[1])).to(device) for x in conversations.queries_tensors]
    print('Targets to gpu...')
    targets = [x.to(device) for x in conversations.targets_tensors]
    print('Training')
    train(features, edges, queries, targets)
    print('\ndone')