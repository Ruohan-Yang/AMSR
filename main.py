import os
import argparse
import torch
from src.Load import pro_data, load_data_s, load_data_c, get_dataloader
from src import AMSR
from src.Train import run_model
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AMSR', help='name of model')
    parser.add_argument('--dataset', type=str, default='Kapferer', help='name of dataset')
    parser.add_argument('--state', type=str, default='layer-wise', help='layer-wise prediction or full_layer prediction')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='numbers of iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train data')
    parser.add_argument('--dim', type=int, default=16, help='dims of node embedding')
    parser.add_argument('--repeats', type=int, default=3, help='numbers of repeats')
    parser.add_argument('--log', type=str, default='./log/', help='record file path')

    args = parser.parse_args()
    path = args.log
    if not os.path.exists(path):
        os.mkdir(path)
    print('The program starts running.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for repeat in range(args.repeats):
        print('***************************')
        args.log = path + args.dataset + '-' + str(args.model) + '.txt'
        print(args)
        log = open(args.log, 'a', encoding='utf-8')
        write_infor = "model:{}, state:{}, lr:{}, epochs:{}, batch:{}, dim:{}".format(args.model, args.state, args.lr, args.epochs, args.batch_size, args.dim)
        log.write('\n\n' + write_infor)
        if args.state == 'layer-wise':
            network_total, layers_pds, node_nums = pro_data(args.dataset)
            # target layer
            for target in range(network_total):
                write_infor = '--- target_network: ' + str(target) + ' ---'
                print(write_infor)
                log.write('\n' + write_infor)
                auxiliary_s = [layer for layer in range(network_total)]
                auxiliary_s.pop(target)
                print('--- auxiliary_networks:', auxiliary_s, '---')
                gcn_data, train, valid, test = load_data_s(target, auxiliary_s, layers_pds, node_nums)
                train, valid, test = get_dataloader(train, valid, test, args.batch_size)
                for i in range(network_total):
                    gcn_data[i].x = gcn_data[i].x.to(device)
                    gcn_data[i].edge_index = gcn_data[i].edge_index.to(device)
                model = eval(args.model).Model_Net(dim=args.dim, layer_number=network_total, gcn_data=gcn_data)
                model = model.to(device)
                run_model(train, valid, test, model, args, device, log)
        elif args.state == 'full_layer':
            network_total, layers_pds, node_nums = pro_data(args.dataset)
            gcn_data, train, valid, test = load_data_c(network_total, layers_pds, node_nums)
            train, valid, test = get_dataloader(train, valid, test, args.batch_size)
            for i in range(network_total):
                gcn_data[i].x = gcn_data[i].x.to(device)
                gcn_data[i].edge_index = gcn_data[i].edge_index.to(device)
            model = eval(args.model).Model_Net(dim=args.dim, layer_number=network_total, gcn_data=gcn_data)
            model = model.to(device)
            run_model(train, valid, test, model, args, device, log)
        else:
            print('Correct predictive state parameters are required.')
            exit(0)
        log.close()