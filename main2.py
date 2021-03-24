import node_sequence as ns
from gensim.models import Word2Vec
import pandas as pd
import load_data as ld
import networkx as nx
import numpy as np
import node2vec as n2v
import argparse
import Evaluation as eval
from scipy.io import loadmat

parser = argparse.ArgumentParser(description="Run link prediction.")
parser.add_argument('--input', nargs='?', default='Data_HPO_AUCn\\test_data2\\network_edges.edgelist',
                    help='Input network file')
parser.add_argument('--mask', default='Data_HPO_AUCn\\test_data2\\mask.mat')
parser.add_argument('--truth', default='Data_HPO_AUCn\\pg_network_use.mat')
parser.add_argument('--output', nargs='?', default='EMB_result\\Deepwalk.embeddings',
                    help='Embeddings path')
parser.add_argument('--Dataset', default='HPO')

##选择哪种方式来进行embedding，默认是node2vec，p=1/q=1 deepwalk
parser.add_argument('--methods', nargs='?', default='node2vec',
                    help='default is node2vec, obtain the vector of each node in the network.')
##node2vec的各种参数
parser.add_argument('--dimensions', type=int, default=128,
                    help='Number of dimensions. Default is 128.')
parser.add_argument('--walk_length', type=int, default=100,
                    help='Length of walk per source. Default is 50.')
parser.add_argument('--num_walks', type=int, default=10,
                    help='Number of walks per source. Default is 10.')
parser.add_argument('--window_size', type=int, default=6,
                    help='Context size for optimization. Default is 10.')
parser.add_argument('--iter', default=1, type=int,
                  help='Number of epochs in SGD')
parser.add_argument('--workers', type=int, default=8,
                    help='Number of parallel workers. Default is 8.')
parser.add_argument('--p', type=float, default=1,
                    help='Return hyperparameter. Default is 1.')
parser.add_argument('--q', type=float, default=1,
                    help='Inout hyperparameter. Default is 1.')
parser.add_argument('--weighted', dest='weighted', action='store_true',
                    help='Boolean specifying (un)weighted. Default is unweighted.')
parser.add_argument('--unweighted', dest='unweighted', action='store_false')
parser.set_defaults(weighted=False)
parser.add_argument('--directed', dest='directed', action='store_true',
                    help='Graph is (un)directed. Default is undirected.')
parser.add_argument('--undirected', dest='undirected', action='store_false')
parser.set_defaults(directed=False)

def n2v_learn_embeddings(args):
    node_walks = ns.node_walk(args)
    node_walks = list(list(map(str, walk)) for walk in node_walks)
    ###生成节点的随机序列
    print('done random walk')
    all_node = set([node for walk in node_walks for node in walk])
    model = Word2Vec(node_walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)
    ###采用skip-gram模型训练，获取节点的embedding
    print('done node embedding')


def read_graph(args):
    if args.weighted:
        g = nx.read_edgelist(args.input, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        g = nx.read_edgelist(args.input, nodetype=str, create_using=nx.DiGraph())
        for edge in g.edges():
            g[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        g = g.to_undirected()
    return g

def node_walk(args):
    nx_g = read_graph(args)
    print("----------")
    g = n2v.Graph(nx_g, args.directed, args.p, args.q)
    print("----------")
    g.preprocess_transition_probs()
    walks = g.simulate_walks(args.num_walks, args.walk_length)
    return walks



args = parser.parse_args(args=[])
node_walks = node_walk(args)
node_walks = list(list(map(str, walk)) for walk in node_walks)
print('done random walk')
all_node = set([node for walk in node_walks for node in walk])
# model = Word2Vec(node_walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers, iter=args.iter)
model = Word2Vec(node_walks, size=args.dimensions, window=args.window_size, min_count=0, workers=args.workers)
model.wv.save_word2vec_format(args.output)
###采用skip-gram模型训练，获取节点的embedding
print('done node embedding')
result_vector = pd.read_table(args.output, header=None, encoding='gb2312', delim_whitespace=True, index_col=0, skiprows=[0])
true_index = result_vector._stat_axis.values.tolist()

if args.Dataset == 'NIPS':
    gnum = 2484
    pnum = 2865

if args.Dataset == 'HPO':
    gnum = 2354
    pnum = 6253

if args.Dataset == 'RDT':
    gnum = 473
    pnum = 758


dimension = args.dimensions
gene_emb = np.zeros(gnum * dimension).reshape(gnum, dimension)
phen_emb = np.zeros(pnum * dimension).reshape(pnum, dimension)
for i in range(1, gnum + 1):
    pair1 = 'g' + str(i)
    if pair1 not in true_index:
        continue
    gi = np.array(result_vector.loc['g' + str(i)])
    gene_emb[i - 1, :] = gi

for i in range(1, pnum + 1):
    pair2 = 'p' + str(i)
    if pair2 not in true_index:
        continue
    pi = np.array(result_vector.loc['p' + str(i)])
    phen_emb[i - 1, :] = pi

pred_mat = phen_emb.dot(gene_emb.T)

# pred_mat = np.asarray(pred_mat)
truth_mat = loadmat(args.truth)
truth_mat = truth_mat["pg_network_use"]

mask_mat = loadmat(args.mask)
mask_mat = mask_mat["mask"]
R_vers = np.ones((pnum, gnum)) - truth_mat
print("药物关联靶蛋白预测：", eval.calculate_metrics_sk(pred_mat, truth_mat, mask_mat + R_vers))
print("药物关联靶蛋白预测：", eval.AUC_main(pred_mat, truth_mat, mask_mat))