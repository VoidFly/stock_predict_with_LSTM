> @李瑶函

## 文件说明

1. 数据处理在main.py中
2. 各项配置在config中
3. 默认采用hs_300数据。可以使用get_data.ipynb处理得到各个特征的变化率数据替换默认数据。

## 模型特点

- 高效、简洁的数据处理逻辑；计算基于numpy，防止python基本数据类型的溢出问题
- 可以设定非对称的涨跌阈值up_threshold和down_threshold
- 各项网络参数可调：隐藏神经元个数、lstm层数等
- 模型训练过程可调：损失函数、优化函数
- 根据环境自动选择GPU训练
- 具有早停机制，快速防止过拟合

## 实验记录

1. 若采取patience=10的早停机制，epoch大约在50左右模型收敛
2. 其他参数待调