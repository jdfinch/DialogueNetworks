
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
        self.att = Attention(features * 2, features * 2)
        self.lin = torch.nn.Linear(features * 2, classes)

    def forward(self, features, edges):
        x = self.conv(features, edges)
        x = self.att(x)
        x = self.lin(x)
        x = torch.nn.functional.softmax(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphInferencer().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(features, edges, targets):
    batch = 10
    model.train()
    running_loss = 0.0
    for i in range(len(features)):
        optimizer.zero_grad()
        feature = features[i]
        edge = edges[i]
        target = targets[i]
        output = model(feature, edge)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % batch == 0:
            print('%5d loss: %.3f' %(i + 1, running_loss / batch))
            running_loss = 0.0


def evaluate(features, edges, targets):
    pass

if __name__ == '__main__':
    print('Features to gpu...')
    features = [x.to(device) for x in conversations.features_tensors]
    print('Edges to gpu...')
    edges = [x.to(device).transpose(0, 1) for x in conversations.edges_tensors]
    print('Targets to gpu...')
    targets = [x.to(device) for x in conversations.targets_tensors]
    print('Training')
    train(features, edges, targets)
    print('\ndone')