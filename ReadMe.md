## 第一次神经网络与深度学习作业
**文件结构**  
.  
├── __init__.py  
├── grid_search_parameter.py  
├── main.py  
├── model_param  
|   └── jhc_module_param.npy  
├── plot_image.py  
├── __pycache__  
|   └── plot_image.cpython-39.pyc  
├── ReadMe.md  
└── test.py  

----

**运行环境**  
pyhton3.9  

----

**文件说明**  
- main.py 为主要的训练文件，直接运行即可，

- test.py 为主要的测试文件，直接运行即可

-  grid_search_parameter.py 用于做超参数搜索的测试文件，直接运行即可

-  plot_image.py 用于画图的相关文件

-  readme.md 使用手册
----
**使用说明**
- 开始训练：需要输入训练的epoch，和用于训练的batch_size: 默认已经是经过参数搜索的超参数
```
pyhton main.py --train_num_epochs = 10 --batch_size = 32
```

- 开始测试：没有超参数，这里采用的是minist的test_dataset,如果没有该数据集会自动下载
```
python test.py
```

- 开始参数搜索
```
python grid_search_parameter.py
```

-----

**模型文件百度云地址**  
链接: https://pan.baidu.com/s/1TnFu7QKx2EqejkdCMks5Cg 提取码: cm6k