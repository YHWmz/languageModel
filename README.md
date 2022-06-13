# 语言模型
## 代码结构
- data：数据文件夹
- log：实验日志文件夹
- modelsave：模型参数文件夹
- tbsave：tensorboard日志文件
- {Model_name}run.sh：对应模型最佳配置的运行脚本
- GPT_run.py：计算预训练模型GPT-2在测试集上的PPL
- ax_tun.py：贝叶斯优化脚本

## 代码运行
分别运行以下脚本，可复现各个模型最佳配置下的性能
```bash
sbatch LSTMrun.sh      
sbatch GRUrun.sh
sbatch RNN_RELU.sh
sbatch RNN_TANH.sh
sbatch Transformerrun.sh
```

