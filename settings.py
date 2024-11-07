city = 'PHO_10cates'  # PHO_mulcates,SIN_10cates,NYC_10cates
gpuId = "cuda"

enable_att2=True
enable_ssl = True
enable_dis = False
mask_prop = 0.1
neg_pos_sample_count = 5
neg_weight = 1
a = 1
b = 1

lr = 1e-4
epoch = 25

if city == 'PHO_10cates' or city == 'NYC_10cates' or city == 'SIN_10cates':
    poi_embed_size = 64
    user_embed_size = 64
    cate_embed_size = 32
    hour_embed_size = 32
    day_embed_size = 32
    run_times = 3

output_file_name = f'Encode_{city}' + "_epoch" + str(epoch) + "_lr" + str(lr)
# 是否对比学习
if enable_ssl:
    output_file_name = output_file_name + "_" + "SSL"
else:
    output_file_name = output_file_name + "_" + "NoSSL"
# 是否在地理距离图上进行注意力
if enable_dis:
    output_file_name = output_file_name + "_" + "diaAtt"
else:
    output_file_name = output_file_name + "_" + "trajAtt"
if enable_att2:
    output_file_name = output_file_name + "_" + "GAT"
else:
    output_file_name = output_file_name + "_" + "NoGAT"

# 日志保存文件名 城市+总epoch数+..
output_file_name = output_file_name + '_embeddingSize' + str(
    poi_embed_size + user_embed_size + cate_embed_size + hour_embed_size + day_embed_size)
