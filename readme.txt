=============================AMH Python Code============================

1.Author: Xianhua Zeng, Jiang Guo, Meng Zhou.
    contact: jiangguo2012@outlook.com

2.We implement AMH on Python2.7.

3.We implement AMH on tensorflow (https://www.tensorflow.org/). Tensorflow version: 1.4.

4.setup: (we use CMGR dataset as a demo(three-modal))
    4.0.before you run AMH tensorflow demo. 
    4.1.preprocessing:
        please download dataset files and pre-trained vgg net file imagenet-vgg-f.mat manually.
            data.rar     https://pan.baidu.com/s/1gpiA3p1krBW1V5_sWd91rQ
    4.2.decompression data.rar to AMH_demo : 
            ./data/dataset  
            ./data/cnnf/imagenet-vgg-f.mat
    4.3.run AMH_demo.py
    4.5.run User_Retrieval.py
5.description:
    AMH_demo.py:           a demo for AMH algorithm on CMGR dataset.
    User_Retrieval.py      A multimodal dataset retrieval example for user.

6.if you have any questions about this demo, please feel free to contact Jiang Guo (jiangguo2012@outlook.com)
    