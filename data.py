import numpy as np
import re
import itertools
import os

def load_data_and_labels():
    # Load data from files
    t1, t2, t3, t4 = [], [], [], []

    with open("converted1.txt") as d:
            l_temp = d.read().split("\n")
            for pair in l_temp:
                l_temp_1 = pair.split(" ")
                t = []
                for num in l_temp_1:
                    if num != '':
                        t.append(int(num))
                if t != []:
                    t1.append(t)

    with open("converted2.txt") as d:
            l_temp = d.read().split("\n")
            for pair in l_temp:
                l_temp_1 = pair.split(" ")
                t = []
                for num in l_temp_1:
                    if num != '':
                        t.append(int(num))
                if t != []:
                    t2.append(t)

    with open("converted3.txt") as d:
            l_temp = d.read().split("\n")
            for pair in l_temp:
                l_temp_1 = pair.split(" ")
                t = []
                for num in l_temp_1:
                    if num != '':
                        t.append(int(num))
                if t != []:
                    t3.append(t)

    with open("converted4.txt") as d:
            l_temp = d.read().split("\n")
            for pair in l_temp:
                l_temp_1 = pair.split(" ")
                t = []
                for num in l_temp_1:
                    if num != '':
                        t.append(int(num))
                if t != []:
                    t4.append(t)

    t1_labels = [[1, 0, 0, 0] for _ in t1]
    t2_labels = [[0, 1, 0, 0] for _ in t2]
    t3_labels = [[0, 0, 1, 0] for _ in t3]
    t4_labels = [[0, 0, 0, 1] for _ in t4]

    documents = t1 + t2 + t3 + t4
    labels = np.concatenate([t1_labels, t2_labels, t3_labels, t4_labels], 0)

    return [documents, labels]

def build_input_data(documents, labels):
    x = np.array(documents)
    y = np.array(labels)

    return [x, y]

def load_data():
    # Load and preprocess data
    documents, labels = load_data_and_labels()
    x, y = build_input_data(documents, labels)
    return [x, y]
