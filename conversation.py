
import networkx as nx
from networkx import MultiDiGraph as Graph
import pickle
import os, shutil, sys
import torch
import numpy as np


class Conversations(list):

    def __init__(self):
        list.__init__(self)
        self.graph = Graph()
        self.concepts = {}
        self.features = []
        self.edgeindices = []
        self.queries = []
        self.target_classes = []
        self.features_tensors = []
        self.edges_tensors = []
        self.queries_tensors = []
        self.targets_tensors = []

    def compile(self):
        print('\nGraph compilation...', end='')
        for i, conversation in enumerate(self):
            conversation.compile()
            for source, target, label in conversation.graph.edges(keys=True):
                self.graph.add_edge(source, target, label)
            if i % 100 == 0:
                print('.', end='', flush=True)
        print()

    def _edges_to_tensors(self, edges, concept_to_features_function):
        features = []
        edge_idices = []
        indices = {}
        for source, target, label in edges:
            for node in (source, target, label):
                if node not in indices:
                    indices[node] = len(indices)
                    features.append(concept_to_features_function(node))
            edge_idices.append([indices[target], indices[label]])
            edge_idices.append([indices[label], indices[source]])
        return features, edge_idices, indices

    def _nodes_to_id(self):
        for edge in self.graph.edges(keys=True):
            for node in edge:
                if node not in self.concepts:
                    self.concepts[node] = len(self.concepts)

    def compile_matrix_data(self, concept_to_features_function):
        self.features = []
        self.edgeindices = []
        self.target_classes = []
        self.queries = []
        self._nodes_to_id()
        for i, conversation in enumerate(self):
            edges_by_turn = [set(t.graph.edges(keys=True)) for t in conversation.turns]
            for j in range(3, len(conversation.turns) - 1):
                context_edges = set().union(*edges_by_turn[:j])
                continuation_edges = set().union(*edges_by_turn[j:j+1]) - context_edges
                features, edge_indices, indices = self._edges_to_tensors(context_edges, concept_to_features_function)
                for source, target, label in continuation_edges:
                    self.queries.append([concept_to_features_function(source), concept_to_features_function(label)])
                    target_class = self.concepts[target]
                    self.features.append(features)
                    self.edgeindices.append(edge_indices)
                    self.target_classes.append(target_class)
            if i % 100 == 0:
                print('.', end='', flush=True)

    def compile_matrices(self):
        self.features_tensors = []
        self.edges_tensors = []
        self.queries_tensors = []
        self.targets_tensors = []
        print('Compiling {} samples to tensors...'.format(len(self.features)))
        for i in range(len(self.features)):
            self.features_tensors.append(torch.from_numpy(np.array(self.features[i])))
            self.edges_tensors.append(torch.from_numpy(np.array(self.edgeindices[i])))
            self.queries_tensors.append(torch.from_numpy(np.array(self.queries[i])))
            self.targets_tensors.append(torch.from_numpy(np.array(self.target_classes[i])))
            if i % 10000 == 0:
                print('.', end='', flush=True)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return True

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Conversation:

    def __init__(self):
        self.text = ''
        self.turns = []
        self.graph = Graph()

    def compile(self):
        for t in self.turns:
            for source, target, label in t.graph.edges(keys=True):
                self.graph.add_edge(source, target, label)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return True

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Turn:

    def __init__(self):
        self.text = ''
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.ner_tags = []
        self.map = {}
        self.graph = Graph()

from itertools import chain
import json
from lark import Lark, Transformer
parser = Lark('''
start: term
term: "(" varname "/" symbol (":" relation (term | value))* ")"
varname: /[a-zA-Z0-9@'_+-]+/
symbol: /[a-zA-Z0-9@'_+-]+/
relation: /[a-zA-Z0-9@'_+-]+/
value: /(?:#[^#]+#|[a-zA-Z0-9@'_&+.-][a-zA-Z0-9@+ &:_'.-]*)/

%import common.WS
%ignore WS
''', parser='lalr')

class AmrTransformer(Transformer):

    relation_instance_count = 0

    def __init__(self, conversation, turn):
        self.graph = Graph()
        Transformer.__init__(self)
        self.conversation = conversation
        self.turn = turn
        self.instances = {}

    def _conceptize(self, string):
        if string == 'amr-unknown':
            string = 'something'
        elif string == '-':
            string = 'not'
        elif string == 'interrogative':
            string = '?'
        elif '-' in string:
            string = string[:string.find('-')]
        elif '' == string:
            string = 'unk'
        string = ''.join([c for c in string if c.isalnum() or c in ' .?!'])
        return string

    def term(self, children):
        varname = children[0].children[0].value
        symbol = children[1].children[0].value
        relationpairs = children[2:]
        for i in range(0, len(relationpairs), 2):
            relation = relationpairs[i].children[0].value
            argument = relationpairs[i+1].children[0].value
            if relation[-3:] == '-of':
                relation = relation[:-3]
                tmp = symbol
                symbol = argument
                argument = tmp
            triple = [self._conceptize(x) for x in (symbol, relation, argument)]
            self.graph.add_edge(triple[2], triple[0], relation, conversation=self.conversation, turn=self.turn)
        return children[1]

erroneous_chars = ['“', '”', '°']
def parse_amr_graph(string, conversation_id, turn_id):
    string = string.replace('#', '@').replace('"', '#')
    for ec in erroneous_chars:
        string = string.replace(ec, 'E')
    parse_tree = parser.parse(string)
    transformer = AmrTransformer(conversation_id, turn_id)
    transformer.transform(parse_tree)
    return transformer.graph

def load_conversations_from(filename, limit=None):
    conversations = Conversations()
    idm = '# ::id '
    sntm = '# ::snt '
    tokm = '# ::tokens ['
    lemm = '# ::lemmas ['
    posm = '# ::pos_tags ['
    nerm = '# ::ner_tags ['
    absm = '# ::abstract_map '
    conversation = Conversation()
    turn = Turn()
    graphstring = ''
    conversation_index = 0
    turn_index = 0
    with open(filename) as file:
        for line in chain(file, ['# ::id 9999999999.0']):
            if idm in line:
                c_idx, t_idx = [int(x.strip()) for x in line[len(idm):].split('.')]
                if c_idx > conversation_index:
                    turn.graph = parse_amr_graph(graphstring, conversation_index, turn_index)
                    conversation.turns.append(turn)
                    graphstring = ''
                    turn = Turn()
                    turn_index = t_idx
                    conversations.append(conversation)
                    conversation_index = c_idx
                    conversation = Conversation()
                    if limit is not None and len(conversations) > limit:
                        break
                    if len(conversations) % 100 == 0:
                        print('...', len(conversations), end='', flush=True)
                elif t_idx > turn_index:
                    turn.graph = parse_amr_graph(graphstring, conversation_index, turn_index)
                    conversation.turns.append(turn)
                    graphstring = ''
                    turn = Turn()
                    turn_index = t_idx
            elif sntm in line:
                turn.text = line[len(sntm):].strip()
            elif tokm in line:
                turn.tokens = [x.replace('"', '').strip() for x in line[len(tokm):-2].split(', ')]
            elif lemm in line:
                turn.tokens = [x.replace('"', '').strip() for x in line[len(lemm):-2].split(', ')]
            elif posm in line:
                turn.pos_tags = [x.replace('"', '').strip() for x in line[len(posm):-2].split(', ')]
            elif nerm in line:
                turn.ner_tags = [x.replace('"', '').strip() for x in line[len(nerm):-2].split(', ')]
            elif absm in line:
                turn.map = json.loads(line[len(absm):])
            elif line:
                graphstring += line #''.join([c for c in line if c.isalpha() or c.isdigit() or c in set(' .+-":()-_/\t\n')])
            else:
                pass
    print('\n', len([c for c in conversations if isinstance(c.graph, Graph)]),
          '/', len(conversations), 'successfully parsed')
    return conversations

import sister
embedder = None
memoization = {}
def fasttext_embed(token):
    global embedder
    if embedder is None:
        embedder = sister.MeanEmbedding(lang="en")
    if token == '':
        token = 'unk'
    if token in memoization:
        return memoization[token]
    memoization[token] = embedder(token)
    return memoization[token]

import cProfile

if __name__ == '__main__':
    convs = load_conversations_from('dailydialog/dd_graphinference_train.txt', limit=3)
    convs.save('conversation_graphs.pckl')
    # convs = Conversations.load('conversation_graphs.pckl')
    convs.compile()
    convs.save('conversation_graphs.pckl')
    print('\nCreating matrix data...\n')
    convs.compile_matrix_data(fasttext_embed)
    convs.save('conversation_graphs.pckl')
    print('Bytes of convs:', sys.getsizeof(convs))
    convs.compile_matrices()
    print('compiled.')
    convs.save('conversation_graphs.pckl')
    print('done')


