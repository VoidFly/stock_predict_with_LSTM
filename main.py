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
    #feature_columns = list(range(1, 5))     # 要作为feature的列,index列不算，第一列数据为第0列
    feature_columns=[1,2,3,5,6]
    label_columns = [4]                  # 要预测的列 close

    up_threshold=0.05
    down_threshold=-0.05

    predict_day = 10             # 预测未来几天

    # 网络参数
    input_size = len(feature_columns)
    
    output_size = 3

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0          # dropout概率
    time_step = 40              # 用前多少天的数据来预测，也是LSTM的time step数

    # 训练参数
    do_train = True
    do_predict = True

    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = True            # 是否使用GPU训练

    train_data_rate = 0.9      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.1      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 5                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42           

    debug_mode = False
    debug_num = 1000  

    used_frame = frame
    model_postfix = {"pytorch": ".pth"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data_path = "./data/choice_hs300.csv"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_save = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False        # 训练loss可视化,采用visdom

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

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step
    
    def read_data(self):
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path,
                                    nrows=self.config.debug_num)
        else:
            init_data = pd.read_csv(self.config.train_data_path)

        return init_data.iloc[:,self.config.feature_columns].to_numpy(), init_data.columns.tolist(), init_data.iloc[:,self.config.label_columns].to_numpy()

    def categorize(self, x):
        if x>self.config.up_threshold:
            return 2
        elif x<self.config.down_threshold: 
            return 0
        else :
            return 1
    def standardize(self,x):
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        return  (x - mean_x)/std_x
        #return (x-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0))


    def get_train_and_valid_data(self):
        feature_data = self.x_data[:self.train_num]
        label_data=self.y_data[:self.train_num+self.config.predict_day]

        rolling_win=[]
        train_x=[]
        train_y=[]
    
        for i in range(self.train_num):
            rolling_win.append(feature_data[i])
            if len(rolling_win)==self.config.time_step:
                train_x.append(rolling_win)
                rolling_win=rolling_win[1:]
                train_y.append((label_data[i+self.config.predict_day]-label_data[i])/label_data[i])


        train_x1=[]
        train_x, train_y = np.array(train_x), np.array(train_y)
        for i in train_x.tolist():
            i=np.array(i)
            i=self.standardize(i)
            train_x1.append(i)

        train_x=np.array(train_x1)
        
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
            
        valid_x1=[]
        valid_x, valid_y = np.array(valid_x), np.array(valid_y)
        for i in valid_x.tolist():
            i=np.array(i)
            i=self.standardize(i)
            valid_x1.append(i)

        valid_x=np.array(valid_x1)
        
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
    #plt.title()
    plt.legend(loc='upper left')

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
