import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os

import csv

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--dataset', default='RetailRocket', help='dataset name: diginetica/Nowplaying/Tmall/RetailRocket/sample')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used') # tsy best performance when the # of layers is 3
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.cuda.set_device(1)

def main():
    #train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    #test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train_sliced.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/train_sliced.txt', 'rb'))

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    elif opt.dataset == 'RetailRocket': # tsy: see the code RetailRocketItemCount.py
         n_node = 36968    
    else:
        n_node = 309
    
    test_data = Data(test_data, shuffle=True, n_node=n_node)
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    
    model = trans_to_cuda(DHCN(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
                
    with open('../results/'+opt.dataset+'_results.csv', 'w', newline='') as results_file:
        csv_writer = csv.writer(results_file)
        
        # Write the header row
        header_row = ['Epoch']
        for K in top_K:
            header_row.extend(['Train Loss', 'Recall@{}'.format(K), 'MRR{}'.format(K), 'Best Epoch for Recall@{}'.format(K), 'Best Epoch for MRR{}'.format(K)])
        csv_writer.writerow(header_row)
        
        # Write results for each epoch
        for epoch in range(opt.epoch):
            row_data = [epoch]
            metrics, total_loss = train_test(model, train_data, test_data)
            for K in top_K:
                metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
                metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
                if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                    best_results['metric%d' % K][0] = metrics['hit%d' % K]
                    best_results['epoch%d' % K][0] = epoch
                if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                    best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                    best_results['epoch%d' % K][1] = epoch
                row_data.extend([total_loss, metrics['hit%d' % K], metrics['mrr%d' % K], best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]])
            csv_writer.writerow(row_data)            

if __name__ == '__main__':
    main()
