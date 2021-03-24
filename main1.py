import argparse
import networkx as nx
import random
import math
from gensim.models import Word2Vec
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.io as io
import sklearn.metrics as sk_metrics
import Evaluation as eval


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="My_method")
#参数设置
# method   HSAEM/ JUST
parser.add_argument('--method', default='HSAEM')
parser.add_argument('--Dataset', default='HPO')
#Data_HPO_AUCn1   6253 * 2354
#Data_HPO_AUCn2   6253 * 2302
parser.add_argument('--input', default='Data_HPO_AUCn2/test_data3/network_edges.txt')
parser.add_argument('--mask', default='Data_HPO_AUCn2/test_data3//mask.mat')
parser.add_argument('--truth', default='Data_HPO_AUCn2/pg_network_use.mat')
# parser.add_argument('--d_depth', default='Data_use/p_depth.mat')
# parser.add_argument('--t_depth', default='Data_use/g_depth_importance.mat')
parser.add_argument('--d_depth', default='Data_use/p_depth.mat')
parser.add_argument('--t_depth', default='Data_use/g_depth_2302gonum.mat')
parser.add_argument('--max_D', default=10)
parser.add_argument('--max_T', default=4)
parser.add_argument('--node_types', default='Data_use/gp_node_types.txt')
parser.add_argument('--output', default='EMB_result/HSAEM.embeddings')
parser.add_argument('--dimensions', type=float, default=128)
#EMAJP-->110  JUST-->100    avarage 100
parser.add_argument('--walk_length', type=int, default=100)
parser.add_argument('--num_walks', type=int, default=10)
parser.add_argument('--window-size', type=int, default=6)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--beta1', type=float, default=0.8) ##表型 -- 药物
parser.add_argument('--beta2', type=float, default=0.9) ##基因 -- 靶蛋白
parser.add_argument('--filter_what', type=str, default='no')
parser.add_argument('--workers', default=10)
#参数初始化
args = parser.parse_args(args=[])
#层次信息
d_depth_mat = loadmat(args.d_depth)
t_depth_mat = loadmat(args.t_depth)


def depth_cal_p(d):
    depth = d_depth_mat["d_depth"]
    depth = depth[0]
    #print(depth)
    return depth[d]

def depth_cal_g(d):
    depth = t_depth_mat["t_depth"]
    depth = depth[0]
    #print(depth)
    return depth[d]

def generate_node_types():
    heterg_dictionary = {'p': ['g'], 'g': ['p']}
    return heterg_dictionary


def gp_generation(G, path_length, heterg_dictionary, filter_what, start):
    path = []
    path_gene = []
    path_phen = []
    return_path = []
    path.append(start)

    cnt = 1
    homog_length = 1
    no_next_types = 0
    heterg_probability = 0

    if start[0] == 'p':
        cur_p = start
    else:
        cur_p = []

    while len(path) < path_length:
        if no_next_types == 1:
            break

        cur = path[-1]
        homog_type = []
        heterg_type = []

        for node_type in heterg_dictionary:
            if cur[0] == node_type:
                homog_type = node_type
                heterg_type = heterg_dictionary[node_type]


        if cur[0] == 'g':
            D = args.max_D
            g_id = int(cur[1:])
            depth_gene = depth_cal_g(g_id - 1)
            # heterg_probability = math.pow(args.beta2, D-depth_gene)
            heterg_probability = math.pow(args.beta2, depth_gene)
            # heterg_probability = 1
            depth_cur = depth_gene


        else:
            D = args.max_T
            phe_id = int(cur[1:])
            depth_phe = depth_cal_p(phe_id - 1)
            #             heterg_probability = math.pow(args.beta1, D-depth_phe)
            C = 1 + np.log2(depth_phe)
            heterg_probability = math.pow(args.beta1, depth_phe)
            #             heterg_probability = math.pow(0.8, depth_phe)
            #             heterg_probability = 0.3
            depth_cur = depth_phe

        if args.method == 'JUST':
            heterg_probability = 1 - math.pow(args.alpha, homog_length)
        if args.method == 'Meta':
            heterg_probability = 1
        if args.method == 'Fix':
            if cur[0] == 'g':
                heterg_probability = args.beta2
            if cur[0] == 'p':
                heterg_probability = args.beta1
        # if args.method == 'Deepwalk':
        #     heterg_probability = random.uniform(0, 1)

        if args.method == 'Com_g':
            if cur[0] == 'g':
                heterg_probability = args.beta2
            else:
                phe_id = int(cur[1:])
                depth_phe = depth_cal_p(phe_id - 1)
                heterg_probability = math.pow(args.beta2, depth_phe)
        #把p的跳转概率设置为定值；对比用层次跳转概率有没有优势
        if args.method == 'Com_p':
            if cur[0] == 'p':
                heterg_probability = args.beta1
                heterg_probability = random.uniform(0, 1)
            else:
                g_id = int(cur[1:])
                depth_gene = depth_cal_g(g_id - 1)
                heterg_probability = math.pow(args.beta2, depth_gene)




        r = random.uniform(0, 1)
        next_type_options = []

        if r <= heterg_probability:
            for heterg_type_iterator in heterg_type:
                next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
            ## 找不到
            if not next_type_options:
                next_type_options = [e for e in G[cur] if (e[0] == homog_type)]

        ## 不跳
        else:
            next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
            # ------------------------------------myself code-------------------------------------------------
            # ------------------------------------gene构造的层次图（在加入下一个可选节点中只添加下层节点）----------------------------------------------
            # if args.method == 'HSAEM':
            #     next_type_options =[]
            #     for e in G[cur]:
            #         if (homog_type =='g') & (e[0] == homog_type):
            #             g_id = int(e[1:])
            #             depth_next = depth_cal_g(g_id-1)
            #             if depth_cur < depth_next:
            #                 next_type_options.append(e)
            #     for e in G[cur]:
            #         if (homog_type =='p') & (e[0] == homog_type):
            #             p_id = int(e[1:])
            #             depth_next = depth_cal_p(p_id-1)
            #             if depth_cur < depth_next:
            #                 next_type_options.append(e)
            # ----------------------------------------------------------------------------------------------------
            # 没有关联的同构节点
            if not next_type_options:
                for heterg_type_iterator in heterg_type:
                    next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])

        if not next_type_options:
            no_next_types = 1
            break

        next_node = random.choice(next_type_options)
        path.append(next_node)
        if next_node[0] == 'g':
            path_gene.append(next_node)
        if next_node[0] == 'p':
            cur_p = next_node
            path_phen.append(next_node)

        if next_node[0] == cur[0]:
            homog_length = homog_length + 1
        else:
            homog_length = 1

    if filter_what == 'no':
        return_path = path
    if filter_what == 'gene':
        return_path = path_gene
    if filter_what == 'phen':
        return_path = path_phen

    return return_path


def generate_walks(G, num_walks, walk_length, filter_what, heterg_dictionary):
    # print('Generating walks .. ')
    walks = []
    nodes = list(G.nodes())

    for cnt in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            just_walks = gp_generation(G, walk_length, heterg_dictionary, filter_what, start=node)
            walks.append(just_walks)
    # print('Walks done .. ')
    return walks

#----------------------主函数部分-------------------------------

# G = nx.read_edgelist(args.input, create_using=nx.DiGraph())
# heterg_dictionary = generate_node_types()
# print("method: ", args.method)
# for d in [4,8,16,32,64,128,256,512]:
#     args.dimensions = d
# # for w in [1,2,3,4,5,6,7,8,9,10]:
# # for a in range(1,10):
#     # args.window_size = w
#     # args.alpah = a/10
#     walks = generate_walks(G, args.num_walks, args.walk_length, args.filter_what, heterg_dictionary)
#     # print('Starting training .. ')
#     model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, workers=args.workers)
#     # print('Finished training .. ')
#     model.wv.save_word2vec_format(args.output)
#     # --------------------------------------------------------------------------
#     result_vector = pd.read_table(args.output, header=None, encoding='gb2312', delim_whitespace=True, index_col=0,
#                                   skiprows=[0])
#     true_index = result_vector._stat_axis.values.tolist()
#
#     if args.Dataset == 'NIPS':
#         gnum = 2484
#         pnum = 2865
#
#     if args.Dataset == 'HPO':
#         gnum = 2354
#         pnum = 6253
#
#     if args.Dataset == 'RDT':
#         gnum = 473
#         pnum = 758
#
#     dimension = args.dimensions
#     gene_emb = np.zeros(gnum * dimension).reshape(gnum, dimension)
#     phen_emb = np.zeros(pnum * dimension).reshape(pnum, dimension)
#     for i in range(1, gnum + 1):
#         #     if i in {42,241,118,135,149,267,366,384,400,412,241}:
#         #         #continue
#         #         i = i + 1
#         pair1 = 'g' + str(i)
#         if pair1 not in true_index:
#             continue
#         gi = np.array(result_vector.loc['g' + str(i)])
#         gene_emb[i - 1, :] = gi
#
#     for i in range(1, pnum + 1):
#         pair2 = 'p' + str(i)
#         if pair2 not in true_index:
#             continue
#         pi = np.array(result_vector.loc['p' + str(i)])
#         phen_emb[i - 1, :] = pi
#     # pred_mat = gene_emb.dot(phen_emb.T)
#     pred_mat = phen_emb.dot(gene_emb.T)
#     # pred_mat = gene_emb.dot(phen_emb.T)
#     # pred_mat = pred_mat.T
#     pred_mat = np.asarray(pred_mat)
#     truth_mat = loadmat(args.truth)
#     truth_mat = truth_mat["gp_network_use"]
#     # mask_mat = loadmat("Data_dot_use/test_data1/mask.mat");
#     mask_mat = loadmat(args.mask)
#     mask_mat = mask_mat["mask"]
#     # print(truth_mat, mask_mat)
#     # pred_mat1= pred_mat.T
#     truth_mat1 = truth_mat.T
#     mask_mat1 = mask_mat
#     # print("window_size=", args.window_size)
#     print("dimensions=", args.dimensions)
#     print("药物关联靶蛋白预测：", calculate_metrics_sk(pred_mat, truth_mat1, mask_mat1))


# G = nx.read_edgelist(args.input, create_using=nx.DiGraph())
# heterg_dictionary = generate_node_types()
# print("method: ", args.method)
# for b1 in range(1,11):
#     for b2 in range(1,11):
#         args.beta1 = b1/10
#         args.beta2 = b2/10
#         walks = generate_walks(G, args.num_walks, args.walk_length, args.filter_what, heterg_dictionary)
#         model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, workers=args.workers)
#         model.wv.save_word2vec_format(args.output)
#         # --------------------------------------------------------------------------
#         result_vector = pd.read_table(args.output, header=None, encoding='gb2312', delim_whitespace=True, index_col=0,
#                                       skiprows=[0])
#         true_index = result_vector._stat_axis.values.tolist()
#         truth_mat = loadmat(args.truth)
#         truth_mat = truth_mat["pg_network_use"]
#         mask_mat = loadmat(args.mask)
#         mask_mat = mask_mat["mask"]
#         pnum, gnum = truth_mat.shape
#
#         dimension = args.dimensions
#         gene_emb = np.zeros(gnum * dimension).reshape(gnum, dimension)
#         phen_emb = np.zeros(pnum * dimension).reshape(pnum, dimension)
#         for i in range(1, gnum + 1):
#             #     if i in {42,241,118,135,149,267,366,384,400,412,241}:
#             #         #continue
#             #         i = i + 1
#             pair1 = 'g' + str(i)
#             if pair1 not in true_index:
#                 continue
#             gi = np.array(result_vector.loc['g' + str(i)])
#             gene_emb[i - 1, :] = gi
#
#         for i in range(1, pnum + 1):
#             pair2 = 'p' + str(i)
#             if pair2 not in true_index:
#                 continue
#             pi = np.array(result_vector.loc['p' + str(i)])
#             phen_emb[i - 1, :] = pi
#
#         pred_mat = phen_emb.dot(gene_emb.T)
#         pred_mat = np.asarray(pred_mat)
#         # R_vers = np.ones((pnum, gnum)) - truth_mat
#         print("beta1,beta2=", args.beta1, args.beta2)
#         print("药物关联靶蛋白预测：", eval.calculate_metrics_sk(pred_mat, truth_mat, mask_mat))
#         print("药物关联靶蛋白预测：", eval.AUC_main(pred_mat, truth_mat, mask_mat))




G = nx.read_edgelist(args.input, create_using=nx.DiGraph())
heterg_dictionary = generate_node_types()
walks = generate_walks(G, args.num_walks, args.walk_length, args.filter_what, heterg_dictionary)
# print('Starting training .. ')
model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, workers=args.workers)
# print('Finished training .. ')
model.wv.save_word2vec_format(args.output)
#--------------------------------------------------------------------------
result_vector = pd.read_table(args.output, header=None, encoding='gb2312', delim_whitespace=True, index_col=0,
                              skiprows=[0])
true_index = result_vector._stat_axis.values.tolist()
truth_mat = loadmat(args.truth)
truth_mat = truth_mat["pg_network_use"]
mask_mat = loadmat(args.mask)
mask_mat = mask_mat["mask"]
pnum, gnum = truth_mat.shape

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

#-------------------------------new AUCn-------------------------------------

pred_mat = np.asarray(pred_mat)
#R_vers = np.ones((pnum, gnum)) - truth_mat
print(args.method)
print("基因关联表型预测：", eval.calculate_metrics_sk(pred_mat, truth_mat, mask_mat))
print("基因关联表型预测：", eval.AUC_main(pred_mat, truth_mat, mask_mat))

