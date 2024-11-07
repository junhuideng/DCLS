import numpy as np
import pandas as pd
import os
def creat_adj(train_valid,nums,city,name):
    adj_matrix = np.zeros((nums, nums))
    poi_count = np.zeros(nums)
    for user in train_valid:
        for sequence in user:#得到该用户的每一条轨迹
            for i in range(len(sequence) - 1):#得到该轨迹长度进行迭代
                for j in range(i + 1, len(sequence)):#得到轨迹中i后面的访问点
                    # 获取两个poi的编号
                    poi1 = sequence[i]
                    poi2 = sequence[j]
                    # 更新邻接矩阵的对称位置，共现次数加一
                    adj_matrix[poi1, poi2] += 1
                    adj_matrix[poi2, poi1] += 1
                poi_count[sequence[i]] += 1
            poi_count[sequence[-1]] += 1
    #print(adj_matrix)
    print(f"###count {name}:",poi_count)
    folder_path = f"./globe_processedDate/{city}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    poi_count_2d = np.stack((np.arange(nums), poi_count),axis=1)
    np.savetxt(f"./globe_processedDate/{city}/{city}_{name}_adjMatrix.csv", adj_matrix, delimiter=",")  # 构造poi的邻接矩阵
    np.savetxt(f"./globe_processedDate/{city}/{city}_{name}_visited_num.csv",  poi_count_2d, delimiter=",", fmt="%d")


def save_indexs(one_mapping,nums,city,name):
    index = np.arange(nums)
    # 你可以使用numpy的column_stack()函数，将index和poi_mapping组合成一个二维数组，赋值给data
    data = np.column_stack((index, one_mapping))
    folder_path = f"./globe_processedDate/{city}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.savetxt(f"./globe_processedDate/{city}/{city}_{name}_index.csv", data, delimiter=",", fmt="%s,%s")

def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def calculate_laplacian_matrix(adj_mat, mat_type):
    from scipy.sparse.linalg import eigsh
    n_vertex = adj_mat.shape[0]
    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')