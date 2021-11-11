# 说明文档

## 一、需要配置
涉及matlab+python联合编程, 生成数据文件`getdata.py`和级联Turbo进行BER测试文件`system_main.py`需要`matlab.engine包`

找到路径`MATLAB\R2017b\extern\engines\python`，执行`python setup.py install`，即可完成安装

**服务器环境**:
```
pytorch 1.7.1
torchvision 0.8.2
python >= 3.6
```

## 二、文件组织
``` 
home
├── data
│   ├── mydataset.py
│   ├── gen_data.npz                                (数据集文件,由getdata.py生成)
│   ├── *.npz                                     (BER记录文件,由system_main.py生成)
│   ├── model_cnn/                                  (模型文件,由system_cnn.py生成)
├── models 
│   ├── CNNNet.py                                       (Encoder和Decoder) 
│   ├── quantization.py                                  (软量化和硬量化)
├── setting 
│   ├── settings.py                                      (随机种子,gpu设置)
├── tools
│   ├── draw_BER.py
│   ├── draw_loss.py
│   ├── logger.py
│   ├── parse.py
├── traditional
│   ├── channel.py
│   ├── match_filtering.py
│   ├── pulse_shaping.py
│   ├── r_filter.py                                  (仿matlab的根号升余弦函数)
├── getdata.py
├── system_cnn.py
├── system_main.py
...
```


## 三、复现流程
### 1. 生成数据
执行
```cmd
python getdata.py
```
生成的数据在`data/gen_data.npz`中  
支持修改的参数有
* train_len: train_len * len
* test_len: test_len * len
* val_len: val_len * len
* ber_len: ber_len * len 为测试BER时总bit数
* len: Turbo编码输入长度6144 bit
* code_len: Turbo编码后长度18444 bit
* train_cut: 截断train_cut生成[train_cut, N]维度的训练集
* test_cut: 截断test_cut生成[test_cut, N]维度的测试集
* val_cut: 截断val_cut生成[val_cut, N]维度的验证集

##### 完整的运行程序:
```cmd
python getdata.py -train_len 150 -test_len 100 -val_len 50 -ber_len 100 -len 6144 -code_len 18444 -train_cut 40000 -test_cut 25000 -val_cut 10000
```







### 2. 训练/测试网络
执行
```cmd
python system_cnn.py
```
生成的模型在`data/model_cnn_4/`(或`data/model_cnn_16`)中

支持修改的参数有
* epoch: 训练周期
* batch_size: 批量大小
* mode: 训练或测试模式
* G: G倍
* K: K倍
* N: 分块长度
* modem_num: 调制数
* unit_T: 软量化函数的T随epoch的增量
* lr_encoder: Encoder学习率
* lr_decoder: Decoder学习率
* lr_step: 学习率衰减周期，每次学习率衰减0.1
* snr: 网络训练时使用的$E_b/n_0$
* snr_start: 网络测试时使用的$E_b/n_0$
* snr_step: 网络测试时使用的$E_b/n_0$
* snr_end: 网络测试时使用的$E_b/n_0$

#### 训练的完整程序:

如下为目前最优训练结果:

##### 2.1. **QPSK**
```cmd
python system_cnn.py -epoch 100 -batch_size 200 -mode train -G 3 -K 20 -N 64 -modem_num 4 -unit_T 5 -lr_encoder 5e-4 -lr_decoder 1e-3 -lr_step 25 -snr 3.0
```
##### 2.2. **16QAM**
```cmd
python system_cnn.py -epoch 100 -batch_size 200 -mode train -G 3 -K 20 -N 64 -modem_num 16 -unit_T 5 -lr_encoder 1e-4 -lr_decoder 1e-3 -lr_step 25 -snr 15.0
```

#### 测试的完整程序:

##### 2.3. **QPSK**
```cmd
python system_cnn.py -batch_size 200 -mode test -G 3 -K 20 -N 64 -modem_num 4 -snr_start -2.0 -snr_step 1.0 -snr_end 10.0 
```
##### 2.4. **16QAM**
```cmd
python system_cnn.py -batch_size 200 -mode test -G 3 -K 20 -N 64 -modem_num 16 -snr_start -2.0 -snr_step 1.0 -snr_end 16.0 
```

#### 目前最优测试结果:
```
对应QPSK, BER范围为[-2.0, 9.0], BER步长为1.0
对应训练snr取3.0，此时BER为0.0952
loss of awgn channel is: [0.6173489894866944, 0.6040156302452088, 0.590239854335785, 0.5765938754081726, 0.562803041934967, 0.5498313927650451, 0.5378839297294616, 0.5276700787544251, 0.5192497591972352, 0.5129511966705322, 0.5086038618087768, 0.5058907380104065]
BER of awgn channel is: [tensor(0.2305, device='cuda:0'), tensor(0.2039, device='cuda:0'), tensor(0.1764, device='cuda:0'), tensor(0.1491, device='cuda:0'), tensor(0.1213, device='cuda:0'), tensor(0.0952, device='cuda:0'), tensor(0.0710, device='cuda:0'), tensor(0.0502, device='cuda:0'), tensor(0.0329, device='cuda:0'), tensor(0.0199, device='cuda:0'), tensor(0.0107, device='cuda:0'), tensor(0.0050, device='cuda:0')]

```

```
对应16QAM, BER范围为[-2.0, 15.0], BER步长为1.0
对应训练snr取15.0，此时BER为0.0878
loss of awgn channel is: [0.6459044361114502, 0.636600911617279, 0.6281255197525024, 0.6193352246284485, 0.6109795231819153, 0.6026459164619445, 0.5946397695541382, 0.5869084000587463, 0.5793593792915345, 0.5728057255744934, 0.5667488703727722, 0.5614229693412781, 0.556223735332489, 0.5523153800964355, 0.5483787531852722,
0.5453689732551574, 0.5429983582496644, 0.541043357849121]
BER of awgn channel is: [tensor(0.2962, device='cuda:0'), tensor(0.2782, device='cuda:0'), tensor(0.2617, device='cuda:0'), tensor(0.2446, device='cuda:0'), tensor(0.2281, device='cuda:0'), tensor(0.2118, device='cuda:0'), tensor(0.1957, device='cuda:0'), tensor(0.1802, device='cuda:0'), tensor(0.1652, device='cuda:0'), tensor(0.1520, device='cuda:0'), tensor(0.1399, device='cuda:0'), tensor(0.1290, device='cuda:0'), tensor(0.1186, device='cuda:0'), tensor(0.1107, device='cuda:0'), tensor(0.1027, device='cuda:0'), tensor(0.0966, device='cuda:0'), tensor(0.0918, device='cuda:0'), tensor(0.0878, device='cuda:0')]

```








### 3. 级联Turbo编码测试BER
执行
```cmd
python system_main.py
```
生成的BER结果在`data/unquantized_4.npz`等中

支持修改的参数有
* curve: 选择绘制有量化、无量化的传统方法，或网络方法
* G: G倍
* K: K倍
* N: 分块长度
* modem_num: 调制数
* snr_start: 网络测试时使用的$E_b/n_0$
* snr_step: 网络测试时使用的$E_b/n_0$
* snr_end: 网络测试时使用的$E_b/n_0$

#### 测试的完整程序:
##### 3.1. **QPSK**

**unquantized**
```cmd
python system_main.py -curve unquantized -G 3 -K 20 -N 64 -modem_num 4 -snr_start -2.0 -snr_step 0.25 -snr_end 2.0
```

**quantized**
```cmd
python system_main.py -curve quantized -G 3 -K 20 -N 64 -modem_num 4 -snr_start -2.0 -snr_step 0.25 -snr_end 3.0
```

**cnn**
```cmd
python system_main.py -curve cnn -G 3 -K 20 -N 64 -modem_num 4 -snr_start -2.0 -snr_step 0.25 -snr_end 2.0
```

##### 3.2. **16QAM**

**unquantized**
```cmd
python system_main.py -curve unquantized -G 3 -K 20 -N 64 -modem_num 16 -snr_start -2.0 -snr_step 0.1 -snr_end 4.0
```

**quantized**
```cmd
python system_main.py -curve quantized -G 3 -K 20 -N 64 -modem_num 16 -snr_start -2.0 -snr_step 0.5 -snr_end 12.0
```

**cnn**
```cmd
python system_main.py -curve cnn -G 3 -K 20 -N 64 -modem_num 16 -snr_start -2.0 -snr_step 0.5 -snr_end 12.0
```