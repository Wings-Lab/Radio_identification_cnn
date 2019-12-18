import numpy as np
import re
import itertools
import os

def load_data_and_labels():
    # Load data from files
    t1, t2, t3, t4 = [], [], [], []
    transmitter_1, transmitter_2, transmitter_3, transmitter_4 = [], [], [], []

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
            i = 0
            temp_list = []
            while i < len(t1):
                temp_list += t1[i]
                i += 1
                if i % 128 == 0 or i == len(t1):
                    temp_list = np.reshape(temp_list, (-1, 2))
                    transmitter_1.append(temp_list)
                    temp_list = []

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
            i = 0
            temp_list = []
            while i < len(t2):
                temp_list += t2[i]
                i += 1
                if i % 128 == 0 or i == len(t2):
                    temp_list = np.reshape(temp_list, (-1, 2))
                    transmitter_2.append(temp_list)
                    temp_list = []

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
            i = 0
            temp_list = []
            while i < len(t3):
                temp_list += t3[i]
                i += 1
                if i % 128 == 0 or i == len(t3):
                    temp_list = np.reshape(temp_list, (-1, 2))
                    transmitter_3.append(temp_list)
                    temp_list = []

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
            i = 0
            temp_list = []
            while i < len(t4):
                temp_list += t4[i]
                i += 1
                if i % 128 == 0 or i == len(t4):
                    temp_list = np.reshape(temp_list, (-1, 2))
                    transmitter_4.append(temp_list)
                    temp_list = []

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
