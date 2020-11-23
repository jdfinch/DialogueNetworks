
import networkx as nx
from networkx import MultiDiGraph as Graph
import pickle
import os, shutil, sys


class Conversations(list):

    def __init__(self):
        list.__init__(self)
        self.graph = Graph()
        self.levi_graph = Graph()

    def compile(self):
        print('\nGraph compilation...', end='')
        for i, conversation in enumerate(self):
            conversation.compile()
            for source, target, label in conversation.graph.edges(keys=True):
                self.graph.add_edge(source, target, label)
            if i % 100 == 0:
                print('.', end='', flush=True)
        print()

    def compile_levi_graphs(self):
        print('\nLevi graph compilation...', end='')
        for i, conversation in enumerate(self):
            conversation.compile_levi_graphs()
            for source, target in conversation.levi_graph.edges():
                self.graph.add_edge(source, target)
            if i % 100 == 0:
                print('.', end='', flush=True)
        print()

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
        self.levi_graph = Graph()

    def compile(self):
        for t in self.turns:
            for source, target, label in t.graph.edges(keys=True):
                self.graph.add_edge(source, target, label)

    def compile_levi_graphs(self):
        for turn in self.turns:
            turn.compile_levi_graph()
            for source, target in turn.levi_graph.edges():
                self.graph.add_edge(source, target)

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
        self.levi_graph = Graph()

    def compile_levi_graph(self):
        self.levi_graph = Graph()
        for source, target, label in self.graph.edges(keys=True):
            self.levi_graph.add_edge(source, label)
            self.levi_graph.add_edge(label, target)


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

def load_conversations_from(filename):
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

if __name__ == '__main__':
    # convs = load_conversations_from('dailydialog/dd_graphinference_train.txt')
    # convs.save('conversation_graphs.pckl')
    convs = Conversations.load('conversation_graphs.pckl')
    # convs.compile()
    convs.compile_levi_graphs()
    convs.save('conversation_graphs.pckl')
    print('done')


