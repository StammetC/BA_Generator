import torch
import os
import numpy as np
import random

from itertools import chain, combinations
from alive_progress import alive_bar

import torch_geometric.data
from PySimpleAutomata import automata_IO
from torch_geometric.data import Data
from string import ascii_lowercase

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

ALPHABET_SIZE = 2

graphical_folder = "graphicalrepresentation"
if not os.path.exists(graphical_folder):
    os.makedirs(graphical_folder)
dataset_folder = "datasets"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


sigma = [i for i in range(ALPHABET_SIZE)]


def connected_states_to_init(kernel):
    edge_index = kernel.edge_index.tolist()
    edge_in = edge_index[0]
    edge_out = edge_index[1]
    connected = [0]
    todo = [0]
    while not len(todo) == 0:
        current_state = todo.pop(0)
        succ_states = [edge_out[i] for i in range(len(edge_out)) if edge_in[i] == current_state]
        for s in succ_states:
            if s not in connected and s not in todo:
                connected.append(s)
                todo.append(s)
    return connected


def tensor_pop(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def remove_state_from_automaton(kernel, state):
    x = kernel.x
    edge_index = kernel.edge_index.tolist()
    edge_in = edge_index[0]
    edge_out = edge_index[1]
    edge_attr = kernel.edge_attr.tolist()

    x = tensor_pop(x, [state])
    for i in reversed(range(len(edge_attr))):
        if edge_in[i] == state or edge_out[i] == state:
            edge_in.pop(i)
            edge_out.pop(i)
            edge_attr.pop(i)
        else:
            if edge_in[i] > state:
                edge_in[i] -= 1
            if edge_out[i] > state:
                edge_out[i] -= 1

    edge_index = torch.tensor([edge_in, edge_out], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.int8)
    x = torch.tensor(x, dtype=torch.float32)
    return Data(x=x, y=kernel.y, edge_index=edge_index, edge_attr=edge_attr)


def prune_automaton(kernel):
    # remove all states not connected to initial state
    # i.e. are never reachable
    unreachable = [s for s in range(len(kernel.x)) if s not in connected_states_to_init(kernel)]
    for state in sorted(unreachable, reverse=True):
        kernel = remove_state_from_automaton(kernel, state)
    # Remove all states with no outgoing transition
    while not len([s for s in range(len(kernel.x)) if s not in kernel.edge_index[0]]) == 0:
        no_out_transition = [s for s in range(len(kernel.x)) if s not in kernel.edge_index[0]]
        for state in sorted(no_out_transition, reverse=True):
            kernel = remove_state_from_automaton(kernel, state)
    return kernel


def draw_graph_from_dot(path: os.path) -> None:
    """
    Takes as input a path to a .dot text representation of an automaton and adds a .svg graphical
    representation of the same automaton to the same folder.

    :param path: path of a .dot text representation of an automaton
    :return: None - adds a graphical .svg representation of the given automaton (as a .dot file)
    """
    folder, file = os.path.split(path)
    nfa_from_file = automata_IO.nfa_dot_importer(path)
    automata_IO.nfa_to_dot(nfa_from_file, file, folder)
    os.remove(path)


def save_automata_from_data(data: torch_geometric.data.Data, filename: str, draw: bool = False) -> None:
    """
    Takes as input a dataelement and a foldername and creates a .dot text representation of
    the given automaton and optionally (boolean 'draw' parameter) also creates a .svg graphical
    representation of the given automaton.
    WARNING: For large automata, this .svg may not be very useful due to readability issues

    :param data: The dataelement to be transformed into a .dot file
    :param filename: The folder where to store the .dot file
    :param draw: If true, also adds a graphical .svg representation
    :return: None - adds a file to the given folder
    """
    src = f"./{graphical_folder}/{filename}.txt"
    file = open(src, "w")
    file.write("digraph{\n")
    file.write("fake [style=invisible]\n")
    number_of_nodes = len(data.x)
    number_of_transitions = len(data.edge_attr)
    number_of_characters = ALPHABET_SIZE
    state_names = []
    char_names = []
    # create all the node names
    for i in range(number_of_nodes):
        name = "q" + str(i)
        state_names.append(name)
    # create all the character names
    for i in range(number_of_characters):
        name = ascii_lowercase[i]
        char_names.append(name)
    char_names.append('m')
    # check for initial state and write to file
    for i in range(number_of_nodes):
        if data.x[i][0] == 1:
            file.write("fake -> ")
            file.write(state_names[i])
            file.write(" [style=bold]\n")
            initial_index = i
    # write all the state names and features to file
    for i in range(number_of_nodes):
        # state is acc and init
        if data.x[i][0] == 1 and data.x[i][1] == 1:
            file.write(state_names[i])
            file.write(" [root=true, shape=doublecircle]\n")
        # state is init
        elif data.x[i][0] == 1:
            file.write(state_names[i])
            file.write(" [root=true]\n")
        # state is acc
        elif data.x[i][1] == 1:
            file.write(state_names[i])
            file.write(" [shape=doublecircle]\n")
        # other states
        else:
            file.write(state_names[i])
            file.write("\n")
    transition_infos = []
    for t in range(number_of_transitions):
        src_index = data.edge_index[0][t]
        dest_index = data.edge_index[1][t]
        if not type(data.edge_attr[t]) == int and False:
            if sum(data.edge_attr[t]) == len(sigma):
                chara_index = len(sigma)
            else:
                chara_index = np.where(data.edge_attr[t].numpy() == 1)[0][0]
        else:
            chara_index = data.edge_attr[t]
        transition_infos.append([src_index, dest_index, [chara_index]])
    to_remove = []
    for i in range(len(transition_infos) - 1):
        for j in range(i+1, len(transition_infos)):
            if (transition_infos[i][0] == transition_infos[j][0]) and (
                    transition_infos[i][1] == transition_infos[j][1]):
                gone = transition_infos[i]
                if i not in to_remove:
                    to_remove.append(i)
                for g in gone[2]:
                    if g not in transition_infos[j][2]:
                        transition_infos[j][2].append(g)
    for index in sorted(to_remove, reverse=True):
        del transition_infos[index]
    for t in transition_infos:
        t[2] = sorted(t[2])
        file.write(state_names[t[0]])
        file.write("->")
        file.write(state_names[t[1]])
        file.write(" [label=\"")
        if len(t[2]) == 1:
            file.write(char_names[t[2][0]])
            file.write("\"]\n")
        else:
            file.write(char_names[t[2][0]])
            for i in range(len(t[2]) - 1):
                file.write(",")
                file.write(char_names[t[2][i + 1]])
            file.write("\"]\n")
    file.write("}")
    file.close()
    if draw:
        draw_graph_from_dot(src)


def save_automata_from_dataset(data: list, foldername: str, draw: bool = False) -> None:
    """
    Takes as input a (slice of a) dataset of automata and a foldername and creates a .dot text representation of
    each given automaton and optionally (boolean 'draw' parameter) also creates a .svg graphical
    representation of the given automata. \n
    WARNING: For large datasets, this will be very time-consuming. Rather use for debugging/testing on small datasets \n
    WARNING: For sets of large automata, this .svg may not be very useful due to readability issues

    :param data: The set of dataelements to be transformed into a .dot file each
    :param foldername: The folder where to store the .dot file
    :param draw: If true, also adds a graphical .svg representation for each automaton
    :return: None - adds the .dot (and optional .svg) files to the given folder
    """
    count = 0
    if not os.path.exists(f"{graphical_folder}/{foldername}"):
        os.makedirs(f"{graphical_folder}/{foldername}")
    with alive_bar(len(data), force_tty=True) as bar:
        for d in data:
            src = foldername + "/" + str(count) + "_" + str(d.y)
            save_automata_from_data(d, src, draw)
            count += 1
            bar()


def generate_adjmatrix_erdosrenyi_automaton(nmin: int, nmax: int, pmin: float,
                              pmax: float, paccmin: float, paccmax: float, prune = True) -> Data:
    """
    Generates one NBW with the given parameters.
    Uses the Erdös-Renyi model for the adjacency matrix,
    then chooses label for each edge from symbol power-set to randomly generate automaton
    Return Tensors are formatted to be given as parameters to torch_geometric.data.Data constructor.

    :param nmin: lower bound of possible number of nodes
    :param nmax: upper bound of possible number of nodes
    :param pmin: lower bound of probability of transition existence
    :param pmax: upper bound of probability of transition existence
    :param paccmin: lower bound of probability of node being accepting
    :param paccmax: upper bound of probability of node being accepting
    :param s: alphabet size
    :param nadd: number of additional node feature vector elements
    :param featinit: how to init add elements: "random", "half" (default) or "zero"
    :return: edge_index, edge_attr and x for Pytorch.data element creation
    """
    n = random.randint(nmin, nmax)
    p = (random.randint(int(100 * pmin), int(100 * pmax))) / 100
    acc_p = (random.randint(int(100 * paccmin), int(100 * paccmax))) / 100

    edge_in = []
    edge_out = []
    edge_attr = []

    # determines what symbols are read by the transition
    # creating power-set of Sigma (excluding empty set)
    possible_transitions = chain.from_iterable(
        combinations(list(sigma), r) for r in range(1, len(sigma) + 1))
    # generate power-set with double single element subsets
    pt = [list(i) for i in possible_transitions] + [[c] for c in sigma]
    # rolls a die for each src, dest state
    # if roll is above probability p, transition is added to automaton
    for src in range(n):
        for dest in range(n):
            diceroll = random.random()
            if diceroll < p:
                # adds transition labels for each one in randomly chosen element in power-set
                for t in random.choice(pt):
                    edge_in.append(src)
                    edge_out.append(dest)
                    edge_attr.append(t)
    # encoding of transitions as arguments for torch.Data constructor
    edge_index = torch.tensor([edge_in, edge_out], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr)

    # Now let's create the node feature vectors
    x = []
    acc_states = []
    # creates feature_vector for each node
    for i in range(n):
        feature_vector = np.zeros(2, dtype=float)
        # first node is initial (by convention)
        if i == 0:
            feature_vector[0] = 1
        else:
            feature_vector[0] = 0
        # diceroll to determine acceptance of state
        diceroll = random.random()
        if diceroll < acc_p:
            feature_vector[1] = 1
            acc_states.append(i)
        else:
            feature_vector[1] = 0
        # adds feature vector to torch.Data input Tensor x
        x.append(feature_vector)
    x = torch.tensor(x, dtype=torch.float32)

    y = torch.tensor([0])

    if prune:
        kernel = prune_automaton(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))
    else:
        kernel = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    if len(kernel.x) == 0:
        return generate_adjmatrix_erdosrenyi_automaton(nmin, nmax, pmin, pmax, paccmin, paccmax)
    else:
        return kernel



nmin = 3
nmax = 9
pmin = 0.2
pmax = 0.8
paccmin = 0.3
paccmax = 0.7


testautomaton = generate_adjmatrix_erdosrenyi_automaton(nmin, nmax, pmin, pmax, paccmin, paccmax)
save_automata_from_data(testautomaton, "testautomaton", True)

testdataset = []
datasetsize = 10
for _ in range(datasetsize):
    testdataset.append(generate_adjmatrix_erdosrenyi_automaton(nmin, nmax, pmin, pmax, paccmin, paccmax))
save_automata_from_dataset(testdataset, "testset", True)

