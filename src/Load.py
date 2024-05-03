import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader

def obtain_sample(inter):
    layer_node_nums = list(set(np.array(inter).reshape(-1)))
    # Consider undirected
    df_extended = pd.DataFrame({'node': inter['pos'], 'pos': inter['node']})
    inter = pd.concat([inter, df_extended], ignore_index=True)
    inter = inter.drop_duplicates().sort_values(by='node').reset_index(drop=True)
    # Negative sample sampling
    group = inter.groupby('node')
    neg_samples = []
    for node, pos_nodes in group:
        pos_list = pos_nodes['pos'].tolist()
        sample_nums = len(pos_list)
        pos_list.append(node)  # Node itself
        if len(layer_node_nums)-sample_nums-1 < sample_nums:
            neg_samples += random.choices(list(filter(lambda x: x not in pos_list, layer_node_nums)), k=sample_nums)
        else:
            neg_samples += random.sample(list(filter(lambda x: x not in pos_list, layer_node_nums)), sample_nums)
    inter['neg'] = neg_samples
    return inter

def obtain_train_edge(inter, target):
    inter = obtain_sample(inter)
    inter['layer'] = target
    # 8:1:1
    df_train = inter.sample(frac=0.8)
    df_temp = inter.drop(df_train.index)
    df_valid = df_temp.sample(frac=0.5)
    df_test = df_temp.drop(df_valid.index)
    return df_train, df_valid, df_test


def gcndata_load(inters, node_nums):
    all_nodes = [i for i in range(node_nums)]
    pos_edge = np.array(inters[['node', 'pos']]).tolist()
    g = nx.Graph(pos_edge)
    g.add_nodes_from(all_nodes)
    adj = nx.to_scipy_sparse_matrix(g, nodelist=all_nodes, dtype=int, format='coo')
    edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))
    x = torch.unsqueeze(torch.FloatTensor(all_nodes), 1)
    gcn_data = Data(x=x, edge_index=edge_index)
    return gcn_data


def pro_data(dataset):
    print('-----------------------------------')
    print('Dataset: ', dataset)
    datadir = 'data/' + dataset + '/'
    layerfiles = os.listdir(datadir)
    network_total = len(layerfiles)
    whole_nodes = []
    for i in range(network_total):
        now_layer = datadir + dataset + str(i) + '.txt'
        now_inter = pd.read_csv(now_layer, sep=' ', header=None)
        now_nodes = list(set(np.array(now_inter).reshape(-1)))
        # print('-----------------------------------')
        # print('Nodes of layer ' + str(i) + ": " + str(len(now_nodes)))
        # print('Edges of layer ' + str(i) + ": " + str(now_inter.shape[0]))
        whole_nodes += now_nodes
    change = list(set(whole_nodes))
    change_dict = {}
    for i in range(len(change)):
        change_dict[change[i]] = i
    whole_node_nums = len(change)
    # print('-----------------------------------')
    # print('Nodes of all layers: ', whole_node_nums)
    # print('-----------------------------------')
    layers_pds = []
    for i in range(network_total):
        layer_path = datadir + dataset + str(i) + '.txt'
        layer = pd.read_csv(layer_path, sep=' ', header=None, names=['node', 'pos'])
        layer['node'] = layer['node'].map(change_dict)  # ID map
        layer['pos'] = layer['pos'].map(change_dict)  # ID map
        layers_pds.append(layer)
    return network_total, layers_pds, whole_node_nums


def load_data_s(tar_layer_id, aux_layer_ids, layers_pds, node_nums):
    # "layer-wise prediction"
    # The training dataset consists of links on the target layer and auxiliary layers
    # The evaluation dataset has only links on the single layer of the target layer
    target_inter = layers_pds[tar_layer_id]
    target_train, target_valid, target_test = obtain_train_edge(target_inter, tar_layer_id)
    gcn_data = {}
    gcn_data[tar_layer_id] = gcndata_load(target_train, node_nums)
    # 辅助层
    aux_inters = []
    for id in aux_layer_ids:
        aux_layer = layers_pds[id]
        aux_layer = obtain_sample(aux_layer)
        aux_layer['layer'] = id
        aux_inters.append(aux_layer)
        gcn_data[id] = gcndata_load(aux_layer, node_nums)
    aux_train = pd.concat(aux_inters, ignore_index=True)
    train = pd.concat([target_train, aux_train], ignore_index=True)
    return gcn_data, train, target_valid, target_test

def load_data_c(network_total, layers_pds, node_nums):
    # "full-layer prediction"
    # the links of all layers are mixed together for prediction
    gcn_data = {}
    train = []
    valid = []
    test = []
    for layer_id in range(network_total):
        layer_inter = layers_pds[layer_id]
        layer_train, layer_valid, layer_test = obtain_train_edge(layer_inter, layer_id)
        train.append(layer_train)
        valid.append(layer_valid)
        test.append(layer_test)
        gcn_data[layer_id] = gcndata_load(layer_train, node_nums)
    train = pd.concat(train, ignore_index=True)
    valid = pd.concat(valid, ignore_index=True)
    test = pd.concat(test, ignore_index=True)
    return gcn_data, train, valid, test

def pro_loader(df):
    pos_inter = df[['node', 'pos', 'layer']]
    pos_inter['link'] = 1
    neg_inter = df[['node', 'neg', 'layer']]
    neg_inter['link'] = 0
    result = np.concatenate((np.array(pos_inter), np.array(neg_inter)), axis=0)
    return result

def pro_dataloader(data, batch_size):
    data = torch.LongTensor(data)
    data_set = TensorDataset(data)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

def get_dataloader(train, valid, test, batch_size):
    train = pro_loader(train)
    train_loader = pro_dataloader(train, batch_size)
    valid = pro_loader(valid)
    valid_loader = pro_dataloader(valid, batch_size)
    test = pro_loader(test)
    test_loader = pro_dataloader(test, batch_size)
    return train_loader, valid_loader, test_loader
