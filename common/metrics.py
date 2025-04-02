import networkx as nx

'''
度
'''

def degree(graph):
    degree_dic = {}
    degree_sum = 0
    degree_list = nx.degree(graph)
    for i in list(degree_list._nodes.keys()):
        degree_sum = degree_sum + degree_list[i]
    for i in list(degree_list._nodes.keys()):
        degree_dic[i] = round(degree_list[i] / degree_sum, 6)
    return degree_dic

'''
介数中心性
'''

def betweeness_centrality(graph):
    betweeness_dic = {}
    betweeness_sum = 0
    # 做一下归一化
    betweeness_list = nx.betweenness_centrality(graph)
    for i in betweeness_list:
        betweeness_sum = betweeness_sum + betweeness_list[i]
        # print(i)
    for i in betweeness_list:
        betweeness_dic[i] = round(betweeness_list[i] / betweeness_sum, 6)
    return betweeness_dic


'''
coreness
'''

def coreness(graph):
    coreness_dic = {}
    coreness_sum = 0
    # 做一下归一化
    coreness_list = nx.core_number(graph)
    for i in list(coreness_list.keys()):
        coreness_sum = coreness_sum + coreness_list[i]
    for i in list(coreness_list.keys()):
        coreness_dic[i] = round(coreness_list[i] / coreness_sum, 6)

    return coreness_dic

'''
特征向量中心性
'''

def eigenector_centrality(graph):
    eigenector_dic = {}
    eigenector_sum = 0
    # 做一下归一化
    eigenector_centrality_list = nx.eigenvector_centrality(graph, max_iter=2000)
    for i in list(eigenector_centrality_list.keys()):
        eigenector_sum = eigenector_sum + eigenector_centrality_list[i]
    for i in list(eigenector_centrality_list.keys()):
        eigenector_dic[i] = round(eigenector_centrality_list[i] / eigenector_sum, 12)
    return eigenector_dic

'''
pagerank
'''

def page_rank(graph):
    return nx.pagerank(graph)