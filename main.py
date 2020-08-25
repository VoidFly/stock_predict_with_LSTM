# -*- coding: UTF-8 -*-
# @ lyh
import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

frame = "pytorch"
from model.model_pytorch import train, predict


class Config:
    # 数据参数
    feature_columns = list(range(1, 5))     # 要作为feature的列,index列不算，第一列数据为第0列
    label_columns = [0]                  # 要预测的列

    up_threshold=0.05
    down_threshold=-0.02

    predict_day = 1             # 预测未来几天

    # 网络参数
    input_size = len(feature_columns)
    
    output_size = 3

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.2          # dropout概率
    time_step = 20              # 用前多少天的数据来预测，也是LSTM的time step数

    # 训练参数
    do_train = True
    do_predict = True

    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = True            # 是否使用GPU训练

    train_data_rate = 0.99      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.01      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 5                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型
    continue_flag = ""           # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练

    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 1000  # 仅用debug_num条数据来调试

    # 框架参数
    used_frame = frame
    model_postfix = {"pytorch": ".pth"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data_path = "./data/hs300_processed.csv"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_save = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False        # 训练loss可视化，pytorch采用visdom

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


class Data:
    def __init__(self, config):
        self.config = config
        self.x_data, self.data_column_name,self.y_data = self.read_data()
        
        self.data_num = self.x_data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        # self.mean = np.mean(self.x_data, axis=0)
        # self.std = np.std(self.x_data, axis=0)
        # self.norm_data = (self.x_data - self.mean)/self.std   # 归一化，去量纲

        #self.mean_y = np.mean(self.y_data, axis=0)
        #self.std_y = np.std(self.y_data, axis=0)
        #self.y_data = (self.y_data - self.mean_y)/self.std_y   # 归一化，去量纲


        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step
    
    # def mean_std(self):
    #     return self.mean,self.std

    def read_data(self):                # 读取初始数据
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path,
                                    nrows=self.config.debug_num,index_col=0)
        else:
            init_data = pd.read_csv(self.config.train_data_path,index_col=0)
        
        return init_data.iloc[:,self.config.feature_columns].to_numpy(), init_data.columns.tolist(), init_data.iloc[:,self.config.label_columns].to_numpy()

    def categorize(self, x):
        if x>self.config.up_threshold:
            return 2
        elif x<self.config.down_threshold: 
            return 0
        else :
            return 1


    def get_train_and_valid_data(self):
        feature_data = self.x_data[:self.train_num]
        label_data=self.y_data[:self.train_num+self.config.predict_day]

        rolling_win=[]
        train_x=[]
        train_y=[]
        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
            #train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            #train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            for i in range(self.train_num):
                rolling_win.append(feature_data[i])
                if len(rolling_win)==self.config.time_step:
                    train_x.append(rolling_win)
                    rolling_win=rolling_win[1:]
                    train_y.append((label_data[i+self.config.predict_day]-label_data[i])/label_data[i])
        else:
            raise('not implemented')

        train_x, train_y = np.array(train_x), np.array(train_y)
        
        mean_x = np.mean(train_x, axis=0)
        std_x = np.std(train_x, axis=0)
        train_x = (train_x - mean_x)/std_x

        mean_y = np.mean(train_y, axis=0)
        std_y = np.std(train_y, axis=0)
        train_y = (train_y - mean_y)/std_y
        #离散分类化

        train_y=np.array([list(map(self.categorize,i)) for i in train_y.tolist()])

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y


    
    def get_test_data(self, return_label_data=False):
        feature_data = self.x_data[self.train_num:]
        label_data=self.y_data[self.train_num:]

        rolling_win=[]
        valid_x=[]
        valid_y=[]
        for i in range(len(feature_data)-self.config.predict_day):
            rolling_win.append(feature_data[i])
            if len(rolling_win)==self.config.time_step: 
                valid_x.append(rolling_win)
                rolling_win=rolling_win[1:]
                valid_y.append((label_data[i+self.config.predict_day]-label_data[i])/label_data[i])
            
        mean_x = np.mean(valid_x, axis=0)
        std_x = np.std(valid_x, axis=0)
        valid_x = (valid_x - mean_x)/std_x

        mean_y = np.mean(valid_y, axis=0)
        std_y = np.std(valid_y, axis=0)
        valid_y = (valid_y - mean_y)/std_y
        valid_y=np.array([list(map(self.categorize,i)) for i in valid_y.tolist()])

        return np.array(valid_x),np.array(valid_y)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)#所有信息输出到控制台

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                  fmt='[ %(asctime)s ] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save:
        file_handler = logging.FileHandler(config.log_save_path + "out.log")
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

def draw(config: Config, origin_data: Data, logger, predict_data: np.ndarray, label_data):
    
    predict_data=[i[0][0] for i in predict_data]
    label_data=[i[0] for i in label_data]

    plt.figure(1)                     # 预测数据绘制
    plt.plot(range(len(label_data)), label_data, label='label')
    plt.plot(range(len(predict_data)), predict_data, label='predict')
    #plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
    plt.legend(loc='upper left')

    # logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
    #         str(np.squeeze(predict_data[-config.predict_day:, i])))
    if config.do_figure_save:
        plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

    plt.show()  

def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = predict(config, test_X)

            pred_result=[np.argwhere(i==max(i))  for i in pred_result]
           

            draw(config, data_gainer, logger, pred_result, test_Y)
    except Exception:
        logger.error("Run Error", exc_info=True)



if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)
