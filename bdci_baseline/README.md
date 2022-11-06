# 目录树:
    ├── data                                                # 数据文件夹
    │──── bdci                                              # 数据集文件夹
    │     ├── train_bdci.json           # 比赛提供的原始数据
    │     ├── rel2id.json               # 关系到id的映射
    │     ├── evalA.json                # A榜评测集
    ├── data_generator.py                                   # 数据处理工具
    ├── merge_result.py                                     # 结果整理
    ├── output                                              # 模型保存路径，k折验证模型  
    │   ├── 1.pth
    │   └── 2.pth
    │   └── 3.pth
    ├── predict.py                                          # 预测代码
    ├── predict.sh                                          # 预测脚本
    ├── pretrain_models                                     # 预训练模型
    ├── requirements.txt                                    # 依赖库
    ├── result                                              # 预测结果集
    ├── res.json                                            # 最终预测结果
    ├── train.py                                            # 训练代码
    ├── data_utils.py                                       # 数据处理类
    ├── model.py                                            # 模型类
    └── util.py                                             # 工具类

# 一、预训练模型下载
    1、链接: https://pan.baidu.com/s/1COlBY1k9yHoGXAZdEpdbWA?pwd=dgre 提取码: dgre
    2、解压后，将预训练模型文件夹放到pretrain_models目录下
    
# 二、安装环境依赖
    安装requirement.txt文件中的依赖包
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --default-time=2000
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-time=2000 tensorflow
    
# 三、生成模型数据
    执行python data_generator.py生成train.json和test.json
    
    
# 四、模型训练
    执行 sh train.sh
    
# 五、模型预测
    执行 sh predict.sh

# 六、注意事项
    1.所用模型：GRTE；github：https://github.com/neukg/GRTE
    2.batch_size不可设置过大。3090卡，显存24G，batch_size为4
