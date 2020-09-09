# -*- coding: UTF-8 -*-
#author 李瑶函
import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
torch.autograd.set_detect_anomaly(True)

class Net(Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=False, dropout=config.dropout_rate)
                         #pytorch rnn 参数顺序默认(seq_len, batch_size, input_size)，设置batch_first=True 变为batch seqlen inputsize
        self.fc1 = Linear(in_features=config.hidden_size, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=3)

    def forward(self, x, hidden=None):
        x=x.permute(1,0,2)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out=lstm_out[-1,:,:]
        #全连接层没有加relu，实验发现加了以后可能正则化太强输出大部分是不涨不跌
        x=(self.fc1(lstm_out))
        x=(self.fc2(x))
        #x=F.softmax(x,dim=2) #后面使用交叉熵损失函数会自动softmax
        
        return x


def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data

    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).long().squeeze()     # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据
    
    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).long().squeeze()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
    model = Net(config).to(device)      # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()                   # pytorch中，训练时要转换成训练模式，启用batchnormalization和dropout；对应还有eval()模式
        train_loss_array = []
        hidden_train = None
         
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            pred_Y = model(_train_X, hidden_train)   
            
            # 交叉熵需要注意 1. target必须是long类型，2. target不能有多余的为1的维度
            # 3. target不是one hot类型，而是列别编码，且为0到c-1

            loss = criterion(pred_Y, _train_Y)  # 计算loss
            optimizer.zero_grad() #将梯度信息置为0
            loss.backward()    # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数

            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:   # 每一百步显示一次
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = model(_valid_X, hidden_valid)

            #if not config.do_continue_train: hidden_valid = None
           
            loss = criterion(pred_Y, _valid_Y)  # 计算loss
             # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))
        if config.do_train_visualized:      # 第一个train_loss_cur太大，导致没有显示在visdom中
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                break


def predict(config, test_X):
    # 获取测试数据
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_Y = model(data_X, hidden_predict)
        
        result = torch.cat((result, pred_Y), dim=0)

    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据
