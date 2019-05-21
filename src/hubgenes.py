#!/usr/bin/env python
import sys
import glob
import os
import collections
import codecs

import string
from collections import defaultdict as dd
import re
import os.path
import math
import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from utils import logger
import functools
import multiprocessing
#gene matrix: the matrix of genes.
def parallelcoef(adjacency,geneMatrix,geneCount,beta,i):
  # print("here") 
  # print (beta)

  if i%100 == 0:
    logger.info("100 passed")
    logger.info("process id: {}".format(multiprocessing.current_process()))
  for j in range(i,geneCount):
    pearson_ij = 0
    if (np.any(geneMatrix[i])==False) | (np.any(geneMatrix[j])==False) | (np.var(geneMatrix[i])==0) | (np.var(geneMatrix[j])==0):
      # print ("here")
      peason_ij = 0
    else:
      pearson_ij = np.corrcoef(geneMatrix[i],geneMatrix[j])[0,1]
    # print(pearson_ij)
    a_ij = pow(abs(pearson_ij),beta)
    adjacency[i][j] = a_ij
    adjacency[j][i] = a_ij  
def geneCoexpression(geneMatrix,beta):
  '''
  geneMatrix: each row: samples for the gene
  unsigned network
  '''

  geneCount = len(geneMatrix)
  sampleCount = len(geneMatrix[0])
  adjacency = np.zeros((geneCount,geneCount))
  # print (geneMatrix)

  # parallel version

  # manager = multiprocessing.Manager()
  # p = multiprocessing.Pool(processes=1)

  # connectivity = zeros(sampleCount,sampleCount)

  # adjacency = zeros(geneCount,geneCount)
  
  # adjacency = manager.list([manager.list([0 for i in range(geneCount)]) for i in range(geneCount)])
  # mono_arg_func = functools.partial(parallelcoef, adjacency,geneMatrix,geneCount,beta)
  for i in range(geneCount):
    if i%100 == 0:
      logger.info("100 passed")
    for j in range(i,geneCount):
      if (np.any(geneMatrix[i])==False) | (np.any(geneMatrix[j])==False) | (np.var(geneMatrix[i])==0) | (np.var(geneMatrix[j])==0):
        # print ("here")
        peason_ij = 0
      else:
        pearson_ij = np.corrcoef(geneMatrix[i],geneMatrix[j])[0,1]
      # print(pearson_ij)
      a_ij = pow(abs(pearson_ij),beta)
      adjacency[i,j] = a_ij
      adjacency[j,i] = a_ij

  # p.map(mono_arg_func, range(geneCount))

  return adjacency
# def parallelTOM(adjacency,tom_dis,node_num,i):

#   if i%100 ==0:
#     logger.info("100 passed")
#   for j in range(i,node_num):
#     if i==j:
#       t_ij =1
#       # tom_dis[j,i]=1
#     else:

#       # for j in range(i,node_num):

#       #exclude i, j

#       # neighbor_i = set([index for index in range(node_num) if adjacency[i][index]>=threshold])-set([i])
#       # neighbor_j = set([index for index in range(node_num) if adjacency[j][index]>=threshold])-set([j])
#       # if j in neighbor_i:
#       #   neighbor_i_j = neighbor_i - set([j])
#       # else:
#       #   neighbor_i_j = neighbor_i 
#       # if i in neighbor_j:
#       #   neighbor_j_i = neighbor_j - set([i])
#       # else:
#       #   neighbor_j_i = neighbor_j 
#       # l_ij = 0

#       # neighbor_ij =  neighbor_i & neighbor_j

#       l_ij = 0
#       for node in range(node_num):
#         if (node!=i) &(node!=j):
#           l_ij += adjacency[i][node]*adjacency[j][node]
#       # l_ij += adjacency[i,j]
#       k_i = sum(adjacency[i])-adjacency[i][i]
#       k_j = sum(adjacency[j])-adjacency[j][j]
#       t_ij_nominator = l_ij+adjacency[i][j]
#       t_ij_denominator = min(k_i,k_j)+1-adjacency[i][j]


#       '''
#       weighted:
#       '''

#       # k_i = sum()
#       # for node in neighbor_i:
#       #   k_i += adjacency[i,node]
#       # k_j = 0
#       # for node in neighbor_j:
#       #   k_j += adjacency[j,node]
#       # t_ij_nominator = l_ij+adjacency[i,j]
#       # t_ij_denominator = min(k_i,k_j)+1-adjacency[i,j]
#       if t_ij_denominator ==0:
#         t_ij = 0
#       else:

#         t_ij = float(t_ij_nominator)/float(t_ij_denominator)
#     t_ij_dis = 1-t_ij
#     tom_dis[i][j]= t_ij_dis
#     tom_dis[j][i]= t_ij_dis

def parallelTOM(adjacency,tom_dis,node_num,i):

  if i%100 ==0:
    logger.info("100 passed")
  for j in range(i,node_num):
    if i==j:
      t_ij =1
      
    else:


      l_ij = 0
      for node in range(node_num):
        if (node!=i) &(node!=j):
          l_ij += adjacency[i][node]*adjacency[j][node]
      # l_ij += adjacency[i,j]
      k_i = sum(adjacency[i])-adjacency[i][i]
      k_j = sum(adjacency[j])-adjacency[j][j]
      t_ij_nominator = l_ij+adjacency[i][j]
      t_ij_denominator = min(k_i,k_j)+1-adjacency[i][j]



      if t_ij_denominator ==0:
        t_ij = 0
      else:

        t_ij = float(t_ij_nominator)/float(t_ij_denominator)
    t_ij_dis = 1-t_ij
    tom_dis[i][j]= t_ij_dis
    tom_dis[j][i]= t_ij_dis

# def calTOM_DIS(adjacency,threshold):
#   node_num = len(adjacency)
#   # tom_dis = np.zeros((node_num,node_num))

#   manager = multiprocessing.Manager()
#   p = multiprocessing.Pool(processes=1)
#   # connectivity = zeros(sampleCount,sampleCount)

#   # adjacency = zeros(geneCount,geneCount)
#   # parallel version
#   tom_dis = manager.list([manager.list([0 for i in range(node_num)]) for i in range(node_num)])
#   mono_arg_func = functools.partial(parallelTOM, adjacency,tom_dis,node_num)
#   p.map(mono_arg_func, range(node_num))

#   return tom_dis

def calTOM_DIS(adjacency,threshold):
  node_num = len(adjacency)
  tom_dis = np.zeros((node_num,node_num))
  for i in range(node_num):
    if i%100 ==0:
      logger.info("100 passed")
    for j in range(i,node_num):
      if i==j:
        t_ij =1
        
      else:


        l_ij = 0
        for node in range(node_num):
          if (node!=i) &(node!=j):
            l_ij += adjacency[i][node]*adjacency[j][node]
        # l_ij += adjacency[i,j]
        k_i = sum(adjacency[i])-adjacency[i][i]
        k_j = sum(adjacency[j])-adjacency[j][j]
        t_ij_nominator = l_ij+adjacency[i][j]
        t_ij_denominator = min(k_i,k_j)+1-adjacency[i][j]



        if t_ij_denominator ==0:
          t_ij = 0
        else:

          t_ij = float(t_ij_nominator)/float(t_ij_denominator)
      t_ij_dis = 1-t_ij
      tom_dis[i][j]= t_ij_dis
      tom_dis[j][i]= t_ij_dis

  return tom_dis

def getHubs(adjacency,tom_dis):
  '''
  Apply hierarchical clustering on the tom_dis and get the clusters, from each cluster, get the highly connected nodes.
  input:
    adjacency: adjacency matrix for all the genes
    tom_dis: topological overlapped dissimilarity matrix
  output: 
    hub_genes: hub genes dictionary, return the hub gene and the nomarlized node connectivity

  '''
  '''
  sklearn
  '''
  model = AgglomerativeClustering(affinity='precomputed', n_clusters=10, linkage='complete').fit(tom_dis)
  labels = model.labels_
  '''
  scipy
  '''
  #hardcode 2 clusters, modify it in the future
  # n_clusters = 2
  # Z = hierarchy.linkage(tom_dis, metric='precomputed')
  # labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
  #plot the dendrogram
  #hierarchy.dendrogram(Z)

  node_cluster = {}
  for node in range(len(labels)):
    if labels[node] in node_cluster:
      node_cluster[labels[node]].append([node,0])
    else:
      node_cluster[labels[node]]= [[node,0]]


  #calculate the node connectivity in each cluster.

  #hard code here, get the top 10 connectivity nodes in each cluster, modify in the future
  
  result_nodes = {}
  print(node_cluster)
  for cluster in node_cluster:
    # total_node_connectivity = 0
    print (cluster)
    num_nodes = len(node_cluster[cluster])
    for node_i in node_cluster[cluster]:
      node_i_connectivity = 0
      for node_j in node_cluster[cluster]:
        if node_j[0]!=node_i[0]:
          node_i_connectivity += adjacency[node_i[0],node_j[0]]
      node_i[1] = node_i_connectivity/num_nodes
    node_cluster[cluster] = sorted(node_cluster[cluster],key=lambda x:x[1],reverse=True)
    top_n = 5
    for each_node in node_cluster[cluster][:top_n]:
      if each_node[0] not in result_nodes:
        result_nodes[each_node[0]] = each_node[1]
  #modified      
  # print(node_cluster)
  # print(result_nodes)
  # return result_nodes
  new_node_cluster = {}
  for each_cluster_key in node_cluster:
    new_node_cluster[each_cluster_key] = []
    for each in node_cluster[each_cluster_key]:
      new_node_cluster[each_cluster_key].append(each[0])
  print("new_node_cluster is:")
  print (new_node_cluster)
  # return new_node_cluster
  return new_node_cluster
    # result_nodes[cluster] = node_cluster[cluster][:10]




def hubSelection(X_train):
  geneMatrix = X_train.transpose()
  # print (geneMatrix)
  logger.info("start adjacency")
  adjacency = geneCoexpression(geneMatrix,2)
  adjacency = np.array(adjacency)
  logger.info("end adjacency")
  np.savetxt('adjacency.out', adjacency, delimiter=',')
  # print(adjacency)
  tom_dis = calTOM_DIS(adjacency,0.5)
  # print (tom_dis)
  tom_dis = np.array(tom_dis)
  np.savetxt('tom.out',tom_dis, delimiter=',')
  # print(adjacency)
  logger.info("tom_dis end")
  # print ("tom_dis end")
  hub_dict = getHubs(adjacency,tom_dis)
  logger.info("start extracting hub")
  
  # feature_index = hub_dict.keys()
  # print (features)

  # return feature_index
  return hub_dict










                




# def main():
 

# '''
# '''

  

# if __name__ == '__main__':
#   main()