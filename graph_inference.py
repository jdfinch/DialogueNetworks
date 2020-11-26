
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric as torchg
from torch_geometric.nn import GATConv
from torch_scatter.composite.softmax import scatter_softmax
from torch_scatter import scatter_add

from conversation import *

conversations = Conversations.load('conversation_graphs_with_prev_flagged.pckl')

feature_size = 302
query_size = 600
batch_size = 1500
classes = len(conversations.concepts)

class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.W = torch.nn.Parameter(torch.rand(self.decoder_dim, self.encoder_dim))

    def forward(self, query, values, index):
        """
        query: torch.Tensor  [batch]
        values: torch.Tensor [nodes, features]
        index: torch.Tensor  [nodes]  (with batch number of labels)
        """
        transformed_values = self.W @ values.T  # [features, nodes]
        attended_values = query @ transformed_values # [batch, nodes]
        attended_values = attended_values / np.sqrt(self.decoder_dim)  # [batch, nodes]
        weights = scatter_softmax(attended_values, index) # [batch, nodes]
        weights = torch.gather(weights, 0, index.unsqueeze(0)).squeeze() # [nodes]
        broadcastable = weights.unsqueeze(0)
        weighted_values = broadcastable * values.T # [features, nodes] (broadcasts weights)
        final_values = scatter_add(weighted_values.T, index, dim=0) # [batch, features]
        return final_values

class GraphInferencer(torch.nn.Module):

    def __init__(self):
        super(GraphInferencer, self).__init__()
        self.conv = GATConv(feature_size, feature_size)
        self.att = Attention(feature_size, query_size)
        self.lin = torch.nn.Linear(feature_size, classes)
        self.ablationlin = torch.nn.Linear(feature_size + query_size, classes)
        self.test1 = torch.nn.Linear(query_size, query_size)
        self.test2 = torch.nn.Linear(query_size, query_size)
        self.test3 = torch.nn.Linear(query_size, classes)
        self.superlin = torch.nn.Linear(query_size + feature_size, feature_size)

    def forward(self, x, edges, query, batch):
        x = self.conv(x, edges)
        x = F.relu(x)
        x = self.conv(x, edges)
        x = F.relu(x)
        x = self.conv(x, edges)
        x = F.relu(x)
        # x = scatter_add(x, batch, dim=0)  # [batch, features]
        # x = torch.cat((x, query), dim=1)
        x = self.att(query, x, batch)
        x = F.relu(x)
        x = self.lin(x)

        # x = self.test1(query)
        # x = F.relu(x)
        # x = self.test2(x)
        # x = F.relu(x)
        # x = self.test3(x)

        # x = torch.nn.functional.softmax(x)
        return x

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = GraphInferencer().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(dataloader, test_dataloader, minitrain, epochs=5):
    # print('initial train MRR:', evaluate(dataloader))
    model.train()
    for e in range(epochs):
        running_loss = 0.0
        print('Epoch', e, '-----------')
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            query = torch.reshape(batch.query, (-1, query_size))
            output = model(batch.x, batch.edge_index, query, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        print('%5d loss: %.3f' % (e, running_loss))
        print('train MRR:', evaluate(minitrain))
        print('test MRR:', evaluate(test_dataloader))

# pred = predicted[i]
# if pred == cls:
#     correct += 1
def evaluate(dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        samples = 0
        rank_total = 0
        for batch in dataloader:
            batch = batch.to(device)
            query = torch.reshape(batch.query, (-1, query_size))
            output = model(batch.x, batch.edge_index, query, batch.batch)
            sortout, indices = torch.sort(output, descending=True)
            predicted = indices[:,0]
            for i in range(len(predicted)):
                cls = batch.y[i]
                rank = ((indices[i] == cls).nonzero()).item()
                samples += 1
                rank_total += (1/(rank+1))
            model.train()
        return rank_total / samples

def create_dataloader(features, edges, queries, targets):
    datas = []
    for i in range(len(features)):
        data = torchg.data.Data(features[i], edges[i], y=targets[i])
        data.query = queries[i]
        datas.append(data)
    dataloader = torchg.data.DataLoader(datas, batch_size=batch_size, shuffle=False)
    return dataloader

def softmax_test():
    w = torch.Tensor([[5, 5, 4, 5, 1],
                      [3, 7, 2, 1, 7]])
    i = torch.LongTensor([0, 0, 1, 1, 1])

    print(scatter_softmax(w, i))

def gather_test():
    w = torch.tensor([[5, 5, 4, 5, 1],
                      [3, 7, 2, 1, 7]], dtype=torch.float)
    i = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    print(torch.gather(w, 0, i.unsqueeze(0)).squeeze())

def scatter_add_test():
    w = torch.tensor([[4, 1, 5],
                      [6, 9, 5],
                      [2, 3, 4],
                      [3, 2, 1],
                      [5, 5, 5]], dtype=torch.float)
    i = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    print(scatter_add(w, i, dim=0))


if __name__ == '__main__':
    print('Features to gpu...')
    features = [x.float() for x in conversations.features_tensors]
    print('Edges to gpu...')
    edges = [x.transpose(0, 1) for x in conversations.edges_tensors]
    print('Queries to gpu...')
    queries = [torch.cat((x[0], x[1])).float() for x in conversations.queries_tensors]
    print('Targets to gpu...')
    targets = [x for x in conversations.targets_tensors]

    mts = int(len(features) * 0.1)
    split = int(len(features) * 0.9)
    train_data = create_dataloader(features[:split], edges[:split], queries[:split], targets[:split])
    test_data = create_dataloader(features[split:], edges[split:], queries[split:], targets[split:])
    minitrain = create_dataloader(features[:mts], edges[:mts], queries[:mts], targets[:mts])

    print('Training')
    train(train_data, test_data, minitrain, epochs=10)


    print('\ndone')