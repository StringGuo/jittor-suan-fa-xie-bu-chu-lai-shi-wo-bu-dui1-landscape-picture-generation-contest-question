# Jittor 草图生成风景比赛 
![2977182477_3e8e35dc4a_b](E:\qqfile\1317724222\filerecv\2977182477_3e8e35dc4a_b-16584984743332.jpg)



## 简介
图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。

本项目包含了第二届计图挑战赛计图 - 草图生成风景比赛的代码实现。本项目的特点是：采用了jittor框架，使用pix2pix的方法对图片处理使生成符合标签含义的风景图片。

## 安装 
本项目可在1张 1080ti 上运行，训练时间约为 12小时。

#### 运行环境
- ubuntu 18.04.6 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

## 数据集

数据集下载链接为https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1

## 训练
单卡训练可运行以下命令：
```
python train.py --output_path ./results --batch_size 16 --data_path ./dataset
```

## 测试
生成测试集上的结果可以运行以下命令：

```
python test.py
```
