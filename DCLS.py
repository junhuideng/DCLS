import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PoiEmbedding(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(PoiEmbedding, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]  # 构建一个 channels 列表，用来表示每一层的通道数
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)
        self.gcn.to(device=device)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)
        return x


# GAT
class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))  # 存储节点特征的线性变换权重313*128
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))  # 存储注意力机制权重 516,1
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)  # 节点特征的线性变换
        e = self._prepare_attentional_mechanism_input(Wh)  # 得到节点之间的注意力得分
        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask
        A = A + 1  # shift from 0-1 to 1-2，这样能够保证A中的0不会影响e的计算
        e = e * A
        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # 节点和注意力权重进行线性组合
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


###

class AttributesEmbedding(nn.Module):
    def __init__(self, user_embed_size, cate_embed_size, hour_embed_size, day_embed_size, vocab_size):
        super().__init__()
        self.user_embed_size = user_embed_size
        self.cate_embed_size = cate_embed_size
        self.hour_embed_size = hour_embed_size
        self.day_embed_size = day_embed_size
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.cat_embed = nn.Embedding(cat_num + 1, self.cate_embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.user_embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.hour_embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.day_embed_size, padding_idx=day_num)

    def forward(self, feature_seq):
        cat_emb = self.cat_embed(feature_seq[1])
        user_emb = self.user_embed(feature_seq[2])
        hour_emb = self.hour_embed(feature_seq[3])
        day_emb = self.day_embed(feature_seq[4])
        return cat_emb, user_emb, hour_emb, day_emb


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq_embedding):
        out = self.dropout(feature_seq_embedding)
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()
        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):  # query=TENSOR(128,)
        q = self.expansion(query)  # Q=TENSOR(352,) # [embed_size]
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # weight=tensor(len,)
        weight = torch.unsqueeze(weight, 1)  # weight=tensor(len,1)
        temp2 = torch.mul(value, weight)  # temp2=tensor(len,352)
        out = torch.sum(temp2, 0)  # out=tensor(352,) # sum([len, embed_size] * [len, 1])  -> [embed_size]
        return out


class DCLS(nn.Module):
    def __init__(
            self,
            vocab_size,
            poiEmb_nfeat,
            poi_embed_size=128,
            user_embed_size=128,
            cate_embed_size=32,
            hour_embed_size=32,
            day_embed_size=32,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size  # 得到各个属性的长度
        self.poi_embed_size = poi_embed_size
        self.total_embed_size = poi_embed_size + user_embed_size + cate_embed_size + hour_embed_size + day_embed_size
        self.PoiEmbedding = PoiEmbedding(ninput=poiEmb_nfeat, nhid=[32, 64], noutput=poi_embed_size, dropout=0.2)

        self.simple_poi_embed = nn.Embedding(vocab_size["poi"] + 1, poi_embed_size, padding_idx=vocab_size["poi"])

        self.embedding = AttributesEmbedding(
            user_embed_size,
            cate_embed_size,
            hour_embed_size,
            day_embed_size,
            vocab_size,
        )
        self.short_encoder = TransformerEncoder(
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.long_encoder = TransformerEncoder(
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )

        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )

        self.final_attention = Attention(
            qdim=user_embed_size,
            kdim=self.total_embed_size
        )
        self.node_attn_model = NodeAttnMap(in_features=poiEmb_nfeat, nhid=poi_embed_size,
                                           use_mask=False)
        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["poi"]))
        self.loss_func = nn.CrossEntropyLoss()

        self.trans_line = nn.Linear(self.total_embed_size, user_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))

    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences:  # each long term sequences
            feature_seq = seq
            seq_len = feature_seq.size(1)
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]

            feature_seq[0, masked_index] = self.vocab_size["poi"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day

            masked_sequences.append(feature_seq)
        return masked_sequences

    def ssl(self, embedding_1, embedding_2, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        def PS_infoNCE_loss_simple(embedding1, embedding2, neg_embedding):  # PHO,SIN SSLLOSS
            pos = score(embedding1, embedding2)
            neg1 = score(embedding1, neg_embedding)
            neg2 = score(embedding2, neg_embedding)
            neg = (neg1 + neg2) / 2

            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss

        def NYC_infoNCE_loss_improved(embedding1, embedding2, neg_embedding, temperature=0.2, scale=32):  # NYC SSLLOSS
            pos = score(embedding1, embedding2) / temperature
            neg = score(embedding1, neg_embedding) / temperature
            neg = torch.sum(neg * score(embedding2, neg_embedding))
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(scale * pos)) - torch.log(
                1e-8 + (1 - torch.sigmoid(scale * neg))))
            return con_loss

        if "NYC" in settings.city:
            ssl_loss = NYC_infoNCE_loss_improved(embedding_1, embedding_2, neg_embedding)
        ssl_loss = PS_infoNCE_loss_simple(embedding_1, embedding_2, neg_embedding)
        return ssl_loss

    def forward(self, sample, neg_sample_list, pos_sample_list, X, A,
                dis_adj_mat):  # neg_sample_list为5条短轨迹 sample为一条长短轨迹对
        if settings.enable_dis:
            dist_matrix = torch.tensor(dis_adj_mat)
            adj_matrix = torch.zeros_like(dist_matrix)
            adj_matrix[dist_matrix <= 1] = 1
            adj_matrix[dist_matrix > 1] = 0
            adj_matrix = adj_matrix.cuda()
            attn_map = self.node_attn_model(X, adj_matrix)

        emodel01 = self.embedding  # 时间，周末，类别，用户的嵌入模型
        emodel02 = self.PoiEmbedding(X, A)  # poi的嵌入模型

        # emodel02=self.simple_poi_embed #poi简易版嵌入模型

        attn_map = self.node_attn_model(X, A)
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[:, :- 1]
        target = short_term_sequence[0, -1]
        user_id = short_term_sequence[2, 0]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences,
                                                settings.mask_prop)

        # Long-term
        long_term_out = []
        for seq in long_term_sequences:
            embedding = self.getSeqembedding(emodel01, emodel02, seq)
            output = self.long_encoder(embedding)
            long_term_out.append(output)
        long_term_state = torch.cat(long_term_out, dim=0)  # 将所有长轨迹的嵌入进行拼接

        # Short-term
        short_embedding = self.getSeqembedding(emodel01, emodel02, short_term_features)
        short_term_state = self.short_encoder(short_embedding)

        user_embed = self.embedding.user_embed(user_id)  # tensor：64
        ##---
        embedding = torch.unsqueeze(short_embedding, 0)
        output, _ = self.lstm(embedding)
        short_term_enhance = torch.squeeze(output)
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.trans_line(
            torch.mean(short_term_enhance, dim=0))

        lstm_output = torch.mean(short_term_enhance, dim=0)
        # SSL长短期编码器对正负样本进行嵌入学习
        neg_long_term_states = []
        for neg_day_sample in neg_sample_list:
            neg_trajectory_features = neg_day_sample
            neg_embedding = self.getSeqembedding(emodel01, emodel02,
                                                 neg_trajectory_features)
            neg_long_term_state = self.long_encoder(
                neg_embedding)
            neg_long_term_state = torch.mean(neg_long_term_state, dim=0)  # tensor(224,)
            neg_long_term_states.append(neg_long_term_state)
        # SSL
        # 短期编码器对正负样本进行嵌入学习
        pos_short_term_states = []  # 短期encoder正样本
        for pos_target_sample in pos_sample_list:
            pos_trajectory_features = pos_target_sample
            pos_embedding = self.getSeqembedding(emodel01, emodel02,
                                                 pos_trajectory_features)
            pos_short_term_state = self.short_encoder(
                pos_embedding)
            pos_short_term_state = torch.mean(pos_short_term_state, dim=0)
            pos_short_term_states.append(pos_short_term_state)

        neg_short_term_states = []  # 短期encoder负样本
        for neg_target_sample in neg_sample_list:
            neg_trajectory_features = neg_target_sample
            neg_embedding = self.getSeqembedding(emodel01, emodel02,
                                                 neg_trajectory_features)
            neg_short_term_state = self.short_encoder(
                neg_embedding)
            neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
            neg_short_term_states.append(neg_short_term_state)
        # -----------------------------------
        # 1、长短编码器之间的对比学习
        short_embed_mean = torch.mean(short_term_state, dim=0)
        long_embed_mean = torch.mean(long_term_state, dim=0)
        neg_embed_mean = torch.mean(torch.stack(neg_long_term_states), dim=0)
        ssl_loss = self.ssl(short_embed_mean, long_embed_mean, neg_embed_mean)
        # 2、短编码器内部的对比学习
        pos_short_mean = torch.mean(torch.stack(pos_short_term_states), dim=0)
        neg_short_mean = torch.mean(torch.stack(neg_short_term_states), dim=0)
        ssl_loss_short = self.ssl(short_embed_mean, pos_short_mean, neg_short_mean)

        # Final predict
        h_all = torch.cat((short_term_state, long_term_state))
        final_att = self.final_attention(user_embed, h_all, h_all)  # final_att=tensor(352,)
        output = self.out_linear(final_att)  # output=tensor(1340,)
        # --------------add----------------------
        adjust_output = self.adjust_pred_prob_by_graph(torch.unsqueeze(output, 0), attn_map, short_term_features)
        label = torch.unsqueeze(target, 0)
        pred_loss = self.loss_func(adjust_output, label)
        if settings.enable_att2 == False:
            raw_output = torch.unsqueeze(output, 0)
            pred_loss = self.loss_func(raw_output, label)
        ##------------
        all_loss = pred_loss + ssl_loss * settings.a + ssl_loss_short * settings.b
        loss = []
        loss.append(all_loss)
        loss.append(pred_loss)  # poi LOSS
        loss.append(ssl_loss)
        loss.append(ssl_loss_short)
        if settings.enable_att2 == False:
            return loss, output  # 12.26
        return loss, torch.squeeze(adjust_output, 0)

    def predict(self, sample, neg_sample_list, post_sample_list, X, A, dis_adj_mat):
        _, pred_raw = self.forward(sample, neg_sample_list, post_sample_list, X, A, dis_adj_mat)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0, -1]

        return ranking, target

    def adjust_pred_prob_by_graph(self, y_pred_poi, attn_map,
                                  short_term_features):  # y_pred_poi=tensor(7,352) attn_map=tensor(1430,1430),short_term_features=tensor(5,7）
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)  # tensor(1,1430)
        for j in range(short_term_features.size()[1]):
            y_pred_poi_adjusted = 0.1 * attn_map[short_term_features[0, j], :] + y_pred_poi

        return y_pred_poi_adjusted

    def getSeqPOIembedding(self, poiembedding, seq):
        poiseq = seq[0].tolist()
        poiembedding_list = []
        for poi in poiseq:
            if poi == self.vocab_size["poi"]:
                poiembed = torch.zeros(self.poi_embed_size).cuda()
                poiembedding_list.append(poiembed)
                continue
            poiembed = poiembedding[poi]
            poiembedding_list.append(poiembed)
        final_tensor = torch.stack(poiembedding_list, dim=0)
        return final_tensor

    def getSeqembedding(self, otherembedmodel, poiembedmodel, seq):
        poi_emb = self.getSeqPOIembedding(poiembedmodel, seq)
        # poi_emb=poiembedmodel(seq[0])
        cat_emb, user_emb, hour_emb, day_emb = otherembedmodel(seq)
        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)
