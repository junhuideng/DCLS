import os
import pickle
import shutil
import pandas as pd
import settings

csv_template_path = f"./results/template.csv"

##清除由某次运行生成的日志，元数据和模型文件
def clear_log_meta_model(output_file_name, run_num):
    """
    Clear the log, meta, and model files generated by this run
    """
    run_name = f"{output_file_name} {run_num}"
    try:
        os.remove(f"./results/{run_name}_log")
    except OSError:
        pass
    try:
        os.remove(f"./results/{run_name}_meta")
    except OSError:
        pass
    try:
        os.remove(f"./results/{run_name}_model")
    except OSError:
        pass


def calculate_average(output_file_name, run_count):
    csv_path = f"./results/{output_file_name}.csv"
    csv_calculate_average(output_file_name, csv_path, run_count)


def csv_calculate_average(output_file_name, file_path, run_count):
    df = pd.read_csv(file_path) #使用 pandas 模块的 read_csv 函数读取 csv 文件，返回一个 DataFrame 对象；
    last_n_rows = df.iloc[-run_count:, :5]#提取最后 run_count 行的前五列
    averages = last_n_rows.mean()#计算每一列的平均值

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        f.write('\n')
        f.write('\n')
        f.write(','.join(averages.astype(str)))
        f.write(f',{output_file_name}')
        f.write('\n')


def print_output_to_file(output_file_name, run_num):
    run_name = f"{output_file_name} {run_num}"
    log_name = f"./results/{run_name}_log"
    file = open(log_name, 'rb')#以二进制模式打开日志文件，创建一个文件对象
    outfile = open(f"./results/{run_name}.txt", 'a')#以追加模式打开文本文件，创建一个文件对象
    epoch_data = {epoch: {1: {}, 5: {}, 10: {}} for epoch in range(settings.epoch)}#用来存储每个 epoch 的评价指标；

    for i in range(4):
        data = pickle.load(file)
        if i == 1:  # recall
            max_local_recall = {1: (0, 0.), 5: (0, 0.), 10: (0, 0.)}
            for epoch, recalls in data.items():
                for k, recall in recalls.items():
                    recall = recall.item()
                    epoch_data[epoch][k]['recall'] = recall
                    if max_local_recall[k][1] < recall:
                        max_local_recall[k] = (epoch, recall)

        elif i == 2:  # ndcg
            max_local_ndcg = {1: (0, 0.), 5: (0, 0.), 10: (0, 0.)}
            for epoch, ndcgs in data.items():
                for k, ndcg in ndcgs.items():
                    ndcg = ndcg.item()
                    epoch_data[epoch][k]['ndcg'] = ndcg
                    if max_local_ndcg[k][1] < ndcg:
                        max_local_ndcg[k] = (epoch, ndcg)
        elif i == 3:  # map
            max_local_map = {1: (0, 0.), 5: (0, 0.), 10: (0, 0.)}
            for epoch, maps in data.items():
                for k, map in maps.items():
                    map = map.item()
                    epoch_data[epoch][k]['map'] = map
                    if max_local_map[k][1] < map:
                        max_local_map[k] = (epoch, map)

    outfile.write(f"{log_name}\n")
    outfile.write(f"recall: {max_local_recall}\n")
    outfile.write(f"ndcg: {max_local_ndcg}\n")
    outfile.write(f"map: {max_local_map}\n")
    outfile.write('---------------------------------------------------\n\n')

    for epoch, data in epoch_data.items():
        outfile.write(f"epoch: {epoch};\n")
        for k, value in data.items():
            outfile.write(f"Recall@{k}: {value['recall']}, NDCG@{k}: {value['ndcg']}, MAP@{k}: {value['map']}\n")
    outfile.write("===================================================")
    outfile.close()
    file.close()

    max_performance = max_local_recall[5][1] + max_local_recall[10][1] + max_local_ndcg[5][1] + max_local_ndcg[10][1]
    #计算最大性能，即 recall@5, recall@10, ndcg@5, ndcg@10 的和
    max_from_different_epochs = [max_local_recall[5][1], max_local_recall[10][1], max_local_ndcg[5][1],
                                 max_local_ndcg[10][1], max_performance, run_name]#存储各种评价指标的最大值，最大性能和运行的名称

    # append the performance to the end of the csv file
    csv_path = f"./results/{output_file_name}.csv"
    if run_num == 1:#是否是第一次运行，如果是，就复制一个模板文件到 csv 文件的路径
        shutil.copyfile(csv_template_path, csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        f.write('\n')
        f.write(','.join([str(x) for x in max_from_different_epochs]))#写入 max_from_different_epochs 列表的内容
