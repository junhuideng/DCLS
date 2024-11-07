import copy
import datetime
import random
import numpy as np
import torch
import pickle
import time
import os
from sklearn.preprocessing import OneHotEncoder
import settings
from DCLS import DCLS
from results.data_reader import print_output_to_file, calculate_average, clear_log_meta_model
from utils import load_graph_adj_mtx, load_graph_node_features, calculate_laplacian_matrix

device = settings.gpuId if torch.cuda.is_available() else 'cpu'
city = settings.city

train_path = "/processed_data/PHO_10cates/PHO_10cates_train_10cates"
valid_path = "/processed_data/PHO_10cates/PHO_10cates_valid_10cates"
mate_path = "/processed_data/PHO_10cates/PHO_10cates_meta_10cates"
POI_adj_mtx = "/globe_processedDate/PHO_10cates/PHO_10cates_poi_adjMatrix.csv"  # poi邻接图
POI_node_feats = "/globe_processedDate/PHO_10cates/PHO_Category_index_merged.csv"  # poi属性
dis_adj_path="/globe_processedDate/PHO_10cates/dist_mat.npy"


def generate_sample_to_device(sample):
    sample_to_device = []
    for seq in sample:
        features = torch.tensor(seq[:5]).to(device)
        sample_to_device.append(features)
    return sample_to_device


def generate_neg_sample_to_device(neg_trajectory):
    features = torch.tensor(neg_trajectory[:5]).to(device)
    day_to_device = features
    return day_to_device


def generate_neg_pos_sample_list(dataset, user_id, target_POI):
    k = settings.neg_pos_sample_count
    # 1、建立负样本列表
    neg_user_sample_to_device_list = []
    neg_user_samples = random.sample(
        [seq[-1] for seq in dataset if seq[0][2][0] != user_id and seq[-1][0][-1] != target_POI],
        k)  # 从用户不一致且最后一个访问点不一致的长短轨迹组中选择其最后一条轨迹作为负样本轨迹
    for neg_user_sample in neg_user_samples:
        neg_user_sample_to_device_list.append(generate_neg_sample_to_device(neg_user_sample))
    # 2、建立正样本列表
    # 从长短轨迹组中选择最后一条并且该条轨迹的终点与目标一致作为正样本
    pos_target_sample_to_device_list = []
    posnum = 0
    pos_target_samples = []
    for seq in dataset:
        for i in range(len(seq)):
            if seq[i][0][-1] == target_POI:
                pos_target_samples.append(seq[i])
                posnum += 1
        if (posnum == k): break

    for pos_target_sample in pos_target_samples:
        pos_target_sample_to_device_list.append(generate_neg_sample_to_device(pos_target_sample))
    return neg_user_sample_to_device_list, pos_target_sample_to_device_list


def feature_mask(seq, mask_prop):
    feature_seq = copy.copy(seq)
    seq_len = len(feature_seq[0])
    mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()  # 根据mask_prop得到mas的数量
    masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
    masked_indexs = masked_index[:mask_count]  # randomly generate mask index随机产生mask index
    for masked_index in masked_indexs:
        feature_seq[0][masked_index] = vocab_size["poi"]  # mask POI
        feature_seq[1][masked_index] = vocab_size["cat"]  # mask cat
        feature_seq[3][masked_index] = vocab_size["hour"]  # mask hour
        feature_seq[4][masked_index] = vocab_size["day"]  # mask day
    return feature_seq


def file_init(run_name, h_params):
    model_path = f"./results/{run_name}_model"
    log_path = f"./results/{run_name}_log"
    meta_path = f"./results/{run_name}_meta"
    print("parameters:", h_params)
    if os.path.isfile(f'./results/{run_name}_model'):
        try:
            os.remove(f"./results/{run_name}_meta")
            os.remove(f"./results/{run_name}_model")
            os.remove(f"./results/{run_name}_log")
        except OSError:
            pass
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()
    return log_path, model_path, meta_path


def train_model(train_set, test_set, h_params, vocab_size, device, run_name):
    torch.cuda.empty_cache()
    log_path, model_path, meta_path = file_init(run_name, h_params)

    # construct model
    rec_model = DCLS(
        vocab_size=vocab_size,
        poiEmb_nfeat=h_params['poiEmb_nfeat'],  # poi嵌入的注意力头数量
        poi_embed_size=h_params['poi_embed_size'],  # poi嵌入
        user_embed_size=h_params['user_embed_size'],  # 用户嵌入
        cate_embed_size=h_params['cate_embed_size'],  # poi类别嵌入
        hour_embed_size=h_params['hour_embed_size'],  # 一天中某一时间段嵌入（24）
        day_embed_size=h_params['day_embed_size'],  # 某一天是否是周末嵌入（0，1）
        num_encoder_layers=h_params['tfp_layer_num'],  # transform中包含EncoderBlock编码器块的个数
        num_lstm_layers=h_params['lstm_layer_num'],  # LSTM层数
        num_heads=h_params['head_num'],  # transform中包含EncoderBlock编码器块中的注意力头个数
        forward_expansion=h_params['expansion'],  # 前馈神经网络的维度扩展（transformer和最后输出linear层公用）
        dropout_p=h_params['dropout']  # 丢失率（transformer和最后输出linear层公用）

    )
    rec_model = rec_model.to(device)
    # 之前训练是否未结束，未结束则继续执行
    start_epoch = 0

    if os.path.isfile(model_path):
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()

        meta_file = open(meta_path, "rb")
        start_epoch = pickle.load(meta_file) + 1
        meta_file.close()

    params = list(rec_model.parameters())
    optimizer = torch.optim.Adam(params, lr=h_params['lr'])
    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}
    for epoch in range(start_epoch, h_params['epoch']):  # h_params['epoch']=25
        begin_time = time.time()
        total_loss = 0.
        poi_loss = 0.
        ssl_loss = 0.
        short_ssl_loss = 0.

        for sample in train_set:  # 一一读取训练集中所有数据进行一轮epoch，每个ample包含长轨迹和短轨迹list,一条轨迹为一个list，一个list包含每个poi点的6个属性即[6,轨迹长度]
            sample_to_device = generate_sample_to_device(sample)  # 将每一个sample数据以(features)形式加载到cuda
            neg_sample_to_device_list = []  # 生成5个负样本轨迹list形式
            if settings.enable_ssl:  # 是否进行对比学习，生成负样本
                user_id = sample[0][2][0]
                target_POI = sample[-1][0][-1]  # 当前poiId
                neg_sample_to_device_list, pos_target_sample_to_device_list = generate_neg_pos_sample_list(train_set,
                                                                                                           user_id,
                                                                                                           target_POI)
            loss, _ = rec_model(sample_to_device, neg_sample_to_device_list, pos_target_sample_to_device_list, X, A,
                                dis_adj_mat)  # 1.4添加距离矩阵
            total_loss += loss[0].detach().cpu()
            poi_loss += loss[1].detach().cpu()
            ssl_loss += loss[2].detach().cpu()
            short_ssl_loss += loss[3].detach().cpu()

            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

        # 一个epoch中训练集全部训练完成后进行测试
        recall, ndcg, map = test_model(test_set, rec_model)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map

        # 记录每个epoch中所有训练数据的平均loss
        avg_loss = total_loss / len(train_set)
        avg_poi_loss = poi_loss / len(train_set)
        avg_ssl_loss = ssl_loss / len(train_set)
        avg_shortssl_loss = short_ssl_loss / len(train_set)
        loss_dict[epoch] = avg_loss
        print(
            f"epoch: {epoch}; average loss: {avg_loss},avg_poi_loss: {avg_poi_loss},avg_ssl_loss: {avg_ssl_loss},avg_shortssl_losss: {avg_shortssl_loss} ,time taken: {int(time.time() - begin_time)}s")
        # 保存模型
        torch.save(rec_model.state_dict(), model_path)
        meta_file = open(meta_path, 'wb')
        pickle.dump(epoch, meta_file)
        meta_file.close()

        # Early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss) > 10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {epoch}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        file.close()


def test_model(test_set, rec_model, ks=[1, 5, 10]):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    for sample in test_set:  # 遍历测试集中每个数据
        sample_to_device = generate_sample_to_device(sample)
        neg_sample_to_device_list = []
        if settings.enable_ssl:
            user_id = sample[0][2][0]
            target_POI = sample[-1][0][-1]  # 当前poiId
            neg_sample_to_device_list, pos_target_sample_to_device_list = generate_neg_pos_sample_list(test_set,
                                                                                                       user_id,
                                                                                                       target_POI)
        pred, label = rec_model.predict(sample_to_device, neg_sample_to_device_list, pos_target_sample_to_device_list,
                                        X, A, dis_adj_mat)  # 1.4
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)
    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")

    return recalls, NDCGs, MAPs


if __name__ == '__main__':
    # 得到当前时间
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Datetime of now：", now_str)

    # 得到模型参数信息
    h_params = {
        'expansion': 4,
        'mask_prop': 0.1,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'loss_delta': 1e-3}
    # 加载训练测试数据
    file = open(train_path, 'rb')
    train_set = pickle.load(file)
    file = open(valid_path, 'rb')
    valid_set = pickle.load(file)
    # 读取各个属性
    file = open(mate_path, 'rb')
    meta = pickle.load(file)
    file.close()
    # 将poi各个属性个数保存到vocab_size
    vocab_size = {"poi": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device)}
    print("the num of pois :", len(meta["POI"]))  # 打印poi个数


    if city == 'PHO_mulcates' or city == 'NYC_mulcates' or city == 'SIN_mulcates':
        h_params['poi_embed_size'] = settings.poi_embed_size
        h_params['user_embed_size'] = settings.user_embed_size
        h_params['cate_embed_size'] = settings.cate_embed_size
        h_params['hour_embed_size'] = settings.hour_embed_size
        h_params['day_embed_size'] = settings.day_embed_size
        h_params['tfp_layer_num'] = 4  # transformer中编码器块的个数
        h_params['lstm_layer_num'] = 2  # LSTM层数
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
    elif city == 'PHO_10cates' or city == 'NYC_10cates' or city == 'SIN_10cates':
        h_params['poi_embed_size'] = settings.poi_embed_size
        h_params['user_embed_size'] = settings.user_embed_size
        h_params['cate_embed_size'] = settings.cate_embed_size
        h_params['hour_embed_size'] = settings.hour_embed_size
        h_params['day_embed_size'] = settings.day_embed_size
        h_params['tfp_layer_num'] = 4
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1

    # 创建模型日志相关信息的文件
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    ### 创建poi轨迹图
    feature1 = 'visitedNum'  # 访问次数
    feature2 = 'categoryId'  # 类别id
    feature3 = 'Latitude'
    feature4 = 'Longitude'
    # 加载邻接图和属性图
    print('loading POI graph...')
    raw_A = load_graph_adj_mtx(POI_adj_mtx)  # 加载poi邻居图 raw_A=ndarry(poinum,poinum)
    raw_X = load_graph_node_features(POI_node_feats,  # 加载poi点特征图 raw_X=ndarry(poinum,4)
                                     feature1,
                                     feature2,
                                     feature3,
                                     feature4)
    num_pois = raw_X.shape[0]
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(
        list(map(lambda x: [x], cat_list))
    ).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    # 打印信息
    print(f"After one hot encoding poi cat, X.shape: {X.shape}")
    print(f"After one hot encoding poi cat, A.shape: {raw_A.shape}")
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=device, dtype=torch.float)
    A = A.to(device=device, dtype=torch.float)
    settings.poiEmb_nfeat = X.shape[1]
    h_params['poiEmb_nfeat'] = X.shape[1]  # 得到poi嵌入维度
    if settings.enable_dis:
        dis_adj_mat = np.load(
            dis_adj_path)  # dis_adj_mat=nadarry(poinum,poinum)
        print(dis_adj_mat.shape)
        print(dis_adj_mat.dtype)
    else:
        dis_adj_mat=None

    print('****************************************start to train****************************************')
    for run_num in range(1, 1 + settings.run_times):
        run_name = f'{settings.output_file_name} {run_num}'
        print(run_name)  # 打印日志相关信息保存的文件名前缀
        train_model(train_set, valid_set, h_params, vocab_size, device, run_name=run_name)
        print_output_to_file(settings.output_file_name, run_num)  # 每一轮run都将相关信息进行保存到txt和csv文件
        print(f"===========================complete the {run_num} training===========================")
        clear_log_meta_model(settings.output_file_name, run_num)  # 删除该轮的log,mate,model
    calculate_average(settings.output_file_name, settings.run_times)  # 在原有的csv日志文件的最后一行添加平均值
