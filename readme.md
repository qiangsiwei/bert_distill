基于BERT的蒸馏实验
================

参考论文《Distilling Task-Specific Knowledge from BERT into Simple Neural Networks》

分别采用keras和pytorch基于textcnn和bilstm(gru)进行了实验

实验数据分割成 1（有标签训练）：8（无标签训练）：1（测试）

在情感2分类clothing的数据集上初步结果如下：

 - 小模型（textcnn & bilstm）准确率在 0.80 ~ 0.81

 - BERT模型 准确率在 0.90 ~ 0.91

 - 蒸馏模型 准确率在 0.87 ~ 0.88

实验结果与论文结论基本一致，与预期相符

后续将尝试其他更有效的蒸馏方案
