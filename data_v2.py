import numpy as np
import re
import itertools
import os, glob

def load_data_and_labels():
    # Load data from files

    t1, t2, t3, t4 = [], [], [], []
    transmitter_1, transmitter_2, transmitter_3, transmitter_4 = [], [], [], []
    num_f = 1

    for file in glob.glob("*.txt"):
        with open(file) as d:
                l_temp = d.read().split("\n")
                for pair in l_temp:
                    l_temp_1 = pair.split(" ")
                    t = []
                    for num in l_temp_1:
                        if num != '':
                            t.append(int(num))
                    if t != []:
                        t1.append(t)
                i = 0
                temp_list = []
                while i < len(t1):
                    temp_list += t1[i]
                    i += 1
                    if i % 128 == 0 or i == len(t1):
                        temp_list = np.reshape(temp_list, (-1, 2))
                        if num_f == 1:
                            transmitter_1.append(temp_list)
                        elif num_f == 2:
                            transmitter_2.append(temp_list)
                        elif num_f == 3:
                            transmitter_3.append(temp_list)
                        elif num_f == 4:
                            transmitter_4.append(temp_list)
                        temp_list = []
        num_f += 1

    t1_labels = [[1, 0, 0, 0] for _ in transmitter_1]
    t2_labels = [[0, 1, 0, 0] for _ in transmitter_2]
    t3_labels = [[0, 0, 1, 0] for _ in transmitter_3]
    t4_labels = [[0, 0, 0, 1] for _ in transmitter_4]

    documents = transmitter_1 + transmitter_2 + transmitter_3 + transmitter_4
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
