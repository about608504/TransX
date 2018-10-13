## TransX论文阅读报告

### 问题

- 给定一个三元组的集合S，其由属于实体集E的实体h,t和属于关系集L的关系l组成，我们的模型要学习实体和关系在低纬度空间中嵌入向量表示。

### 所需要的背景知识

- TensorFlow常用函数和基本概念

  https://www.cnblogs.com/wuzhitj/p/6648641.html

- Word2vec的基本概念

  https://zhuanlan.zhihu.com/p/26306795

### 提出动机

- 层次的关系在知识数据库中非常常见，翻译是来表示它们的自然变化。
- 从自由文本中学习词嵌入，以及不同类型的实体之间的一对一关系，比如国家与城市之间的”首都”被模型表示为嵌入空间中的翻译。这表明可能存在某种嵌入空间，其中不同类型的实体之间的一对一关系也可以由翻译表示。而TransX的目的就是要强化这种嵌入空间的结构。

### 核心思想

关系在嵌入空间中表示为翻译，即给定一个三元组(h,l,t)，那么尾实体t的嵌入向量应该接近(并且是最接近)于头实体h的嵌入向量与依赖关系l的某向量之和。若三元组不存在，则尾实体的嵌入向量应该远离后两者之和。 想要得到类似这样的效果V (king) − V (queen) ≈ V (man) − V (woman)

- 具体过程

以TransE为例，与之前的基于能量的框架一样，一个三元组的能量用来表示，d()即损失函数，可以采取L1距离(曼哈顿距离)或者L2距离(欧几里得距离)。该能量函数通过h+l与t的距离来表示该三元组的置信度，函数值越小，则该三元组置信度越高。
$$
d(h+l,t) = |l_{h+l}-l_t|_{L1/L2}
$$

- 优化目标函数为

$$
\Gamma = \sum_{(h,l,t)\in S} \sum_{(h^{'},l,t^{'})\in S^{'}}[\gamma+d(h+l,t)-d(h^{'}+l,t^{'})]_{+}
$$

- [x]+取的是x的正数部分。
- S是合法三元组的集合，而S’是错误(corrupted)三元组的集合，我们通过将S中每个三元组的h或t随机替换成一个实体构成，注意不是同时替换。

$$
S^{'}_{(h,l,t)}=\{(h^{'},l,t)|h^{'}\in E\}\bigcup\{(h,l,t^{'})|t^{'}\in E\}
$$

- γ是大于0的先验参数(hyperparameter)，表示的是合法三元组和错误的三元组之间的间隔距离。

- 实体以及关系的初始化采用了Xavier初始化。
- 使用随机梯度下降算法优化目标函数，使用batch的方式。从目标函数也很好推断出算法的嵌套运行，先从训练集选择出一个三元组的集合，对于每一个这样的三元组，采样出一个错误的三元组，然后通过恒定的学习速率的梯度步骤更新参数。算法基于验证集的性能停止。

对于该优化目标函数，个人理解是合法三元组的能量函数值是要小于错误三元组的，我们希望通过在训练集的训练过程中，使每个合法三元组的能量函数值与错误三元组的能量函数值的差值尽可能小于设定好的先验参数γ，从而得到实体和关系的嵌入向量。

![image-20181005195843020](https://ws2.sinaimg.cn/large/006tNbRwly1fvxm9z4xgfj31ae0fk77f.jpg)

### 几种常用的Trans模型简介

- TransE、TransH、TransR

  参考刘知远《知识表示学习研究进展》

  http://nlp.csai.tsinghua.edu.cn/~lyk/publications/knowledge_2016.pdf

### 学习TransX模型可以进行的一些后续工作

以TransE为代表的知识表示学习模型是在原有的知识库上建模，而原有的知识库在很多情况下都是不完备的，这样会导致原有的一些模型无法处理zero-shot等问题，所以引入外部知识对知识表示学习模型进行扩展，增强模型的语义表达能力是很有必要的，几篇综述文章里都提到了在TransE等模型的基础上加入外部知识，比如加文本，加规则，加路径等等，除了加入额外信息，还有研究人员提出用张量（Tensor）等方法来进行表示学习的建模，知识表示学习模型总体上可分为两大类，一类是以TransE为代表的不加额外信息的模型，另一类是加入了各种信息的模型，当然加入的额外信息不止这几类，文件夹里的论文也只是稍微列举一些代表，各种模型五花八门层出不穷，如果感兴趣可以适当了解一下。

#### 综述

- Representation Learning A Review and New Perspectives

  https://ieeexplore.ieee.org/document/6472238/

- Knowledge Graph Embedding A Survey of Approaches and Applications

  https://ieeexplore.ieee.org/document/8047276

- An overview of embedding models of entities and relationship knowledge base completion

  https://arxiv.org/abs/1703.08098

#### 结合图像

- Image-embodied Knowledge Representation Learning

  https://www.ijcai.org/proceedings/2017/0438.pdf

#### 结合规则

- Knowledge Base Completion Using Embeddings and Rules

  https://www.ijcai.org/Proceedings/15/Papers/264.pdf

- Knowledge Graph Embedding with Iterative Guidance from Soft Rules

  https://arxiv.org/abs/1711.11231

#### 结合路径

- DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning

  https://arxiv.org/abs/1707.06690

- Modeling Relation Paths for Representation Learning of Knowledge Bases

  https://arxiv.org/abs/1506.00379

#### 结合文本

- Joint Representation Learning of Text and Knowledge for Knowledge Graph Completion

  https://arxiv.org/abs/1611.04125

- Chains of Reasoning over Entities Relations and Text using Recurrent Neural Networks

  https://arxiv.org/abs/1607.01426

- Compositional Learning of Embeddings for Relation Paths in Knowledge Bases and Text

  http://www.aclweb.org/anthology/P16-1136

#### 张量表示

- Blending of Two and Three-way Interactions for Modeling Multi-relational Data

  https://hal.archives-ouvertes.fr/hal-01131957/document

- Multi-relational Learning Using Weighted Tensor Decomposition with Modular Loss

  https://arxiv.org/abs/1303.1733

# TransX实验报告

### 1.可供参考学习的代码资源

- OpenKE开源项目：

  https://github.com/thunlp/openke

- 完全使用TensorFlow实现的一个TransE版本：

  https://github.com/ZichaoHuang/TransE/blob/master/src/model.py

- 基于OpenKE的TransX简略版本：

  https://github.com/thunlp/TensorFlow-TransX

### 2.实验环境

Pycharm+python3.6+TensorFlow1.4+i5（1000次TransE训练需要40分钟）

Ubuntu14.04+python3.6+TensorFlow1.4+1080*2（1000次TransE训练需要6分钟）

### 3.实验步骤

首先将代码从git上面下载下来，以基于OpenKE的TransX简略版本为例

```bash
git clone https://github.com/thunlp/TensorFlow-TransX
cd TensorFlow-TransX
bash make.sh
```

该项目提供了三种模式

①参数随机初始化的训练

1. Change class Config in transX.py

   class Config(object):

   ```python
   	def __init__(self):
   		...
   		lib.setInPath("your training data path...")
   		self.testFlag = False
   		self.loadFromData = False
   		...
   ```

2. python transX.py

②读取预训练模型继续进行训练

1. Change class Config in transX.py

   class Config(object):

   ```python
   	def __init__(self):
   		...
   		lib.setInPath("your training data path...")
   		self.testFlag = False
   		self.loadFromData = True
   		...
   ```

2. python transX.py

③读取训练完的模型，并进行测试

1. Change class Config in transX.py

   class Config(object):

   ```python
   	def __init__(self):
   		...
   		test_lib.setInPath("your testing data path...")
   		self.testFlag = True
   		self.loadFromData = True
   		...
   ```

2. python transX.py

### 关于实验的一些说明

开始训练之后，便会打印出训练数据的数据集路径，并会不停输出训练得到的loss值

![image-20181005112011450](https://ws2.sinaimg.cn/large/006tNbRwly1fvxm9lupjfj30se0ggjtp.jpg)

完成训练后会在目录中出现model.vec.data-00000-of-00001，model.vec.index，model.vec.meta这三个文件，建议先把训练轮次调整到10-50，看到实验结果可以正常保存之后，再读取之前训练得到的数据去继续接下去的训练。

### 数据集介绍

在训练测试过程中主要使用entity2id.txt，relation2id.txt，train2id.txt，valid2id.txt，test2id.txt

- entity2id.txt:实体编码文件，从 0 开始为每个实体分配一个 ID，第一行是实体的总个数，从第 二行开始每行表示一个实体及其 ID，以 Tab 分隔，如下所示。 

```
  14951
  /m/027rn	0
  /m/06cx9	1
  /m/017dcd	2
  /m/06v8s0	3
  /m/07s9rl0	4
  /m/0170z3	5
  /m/01sl1q	6
  /m/044mz_	7
  /m/0cnk2q	8
```

- relation2id.txt:关系编码文件，与实体编码文件格式相似。 

- train2id.txt:训练数据文件，数据中的所有实体与关系均用ID表示，第一行是数据总个数，从第 二行开始每行表示一个三元组，以 Tab 分隔，如下所示。注意，这里每行的三元组表示为 (h, t, r)， 前两个是实体，最后一个是关系。 

  ```
  483142
  0 1 0
  2 3 1
  4 5 2
  6 7 3
  8 9 4
  10 11 5
  12 13 6
  14 15 7
  16 17 8
  ```

- valid2id.txt:验证数据文件，格式与训练数据类似。 

- test2id.txt:测试数据文件，格式与训练数据类似。 

同时会一些1-n，n-n的关系，也做了相应的测试集，来评价模型的准确率。

### 模型评价分析

##### 基本的评价过程

假设整个知识库中一共有n个实体，那么评价过程如下：

- 将一个正确的三元组a中的头实体或者尾实体，依次替换为整个知识库中的所有其它实体，也就是会产生n个三元组。
- 分别对上述n个三元组计算其能量值，在transE中，就是计算h+r-t的值。这样可以得到n个能量值，分别对应上述n个三元组。
- 对上述n个能量值进行升序排序。
- 记录三元组a的能量值排序后的序号。
- 对所有的正确的三元组重复上述过程。
- 每个正确三元组的能量值排序后的序号求平均，得到的值我们称为Mean Rank。
- 计算正确三元组的能量排序后的序号小于10的比例，得到的值我们称为Hits@10。

有两个指标：Mean Rank和Hits@10。其中Mean Rank越小越好，Hits@10越大越好。

##### 改进

上述过程存在一个不合理的地方：在将一个正确的三元组a的头或者尾实体替换成其它实体之后得到的这个三元组也有可能是正确的，在计算每个三元组的能量并排序之后，这类正确的三元组的能量有可能排在三元组a的前面。但是上面所说的基本评价过程并没有考虑这点。因此我们把上述基本评价过程得到的结果称为Raw Mean Rank和Raw Hits@10，把改进方法得到的结果称为Filtered Mean Rank和Filtered Hits@10。

为了更好的评价embedding的质量，我们对上述方法进行改进。

- 将一个正确的三元组a中的头实体或者尾实体，依次替换为整个知识库中的所有其它实体，也就是会产生n个三元组。
- 分别对上述n个三元组计算其能量值，在transE中，就是计算h+r-t的值。这样可以得到n个能量值，分别对应上述n个三元组。
- 对上述n个能量值进行升序排序。
- 记录三元组a的能量值排序后的序号k。
- 如果前k-1个能量对应的三元组中有m个三元组也是正确的，那么三元组a的序号改为k-m。
- 对所有的正确的三元组重复上述过程。
- 每个正确三元组的能量值排序后的序号求平均，得到的值我们称为Filtered Mean Rank。
- 计算正确三元组的能量排序后的序号小于10的比例，得到的值我们称为Filtered Hits@10

### FAQ：

Q：为什么在读入路径的时候只能读到最前面的.

A：试着将完整的路径写在init.cpp和test.cpp中，make之后再执行

Q：为什么代码汇总print语法会报错

A：由于python版本的小差异，将print修改成print()就可以了

Q：为什么最后训练的模型无法进行保存

A：需要在model前面加上./不然无法识别到正确的目录

Q：为什么我训练得到的TransR模型各方面准确率都低于TransE

A：由于TransR模型具有n个二维的映射矩阵，参数众多，所以训练起来更加耗时，可以试着多训练一段时间，让loss值收敛。

```python
if not config.testFlag:
   for times in range(config.trainTimes):
      res = 0.0
      for batch in range(config.nbatches):
         lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config.batch_size)
         res += train_step(ph, pt, pr, nh, nt, nr)
         current_step = tf.train.global_step(sess, global_step)
      print(times)
      print(res)
   saver.save(sess, './model.vec')
```

| TransE results: |  Mean Rank  | Hits@10  |  Hits@3  |  Hits@1  |
| :-------------: | :---------: | :------: | :------: | :------: |
|      left       | 273.326141  | 0.463408 | 0.254253 | 0.107565 |
|  left(filter)   |  88.823875  | 0.729749 | 0.559632 | 0.331533 |
|      right      | 169.436615  | 0.544176 | 0.318075 | 0.15263  |
|  right(filter)  |  54.520813  | 0.785868 | 0.615429 | 0.378341 |
|     results     |             |          |          |          |
|      left       |  222.54155  | 0.477595 | 0.270234 | 0.12656  |
|  left(filter)   |  38.039276  |  0.7632  | 0.601497 | 0.385638 |
|      right      | 136.306992  | 0.55826  | 0.334834 | 0.175535 |
|  right(filter)  |  21.39119   | 0.810127 | 0.647526 | 0.422847 |
|  results(1-1):  |             |          |          |          |
|      left       |  196.62175  | 0.769504 | 0.602837 | 0.322695 |
|  left(filter)   | 196.356979  | 0.774232 | 0.626478 | 0.336879 |
|      right      | 223.081558  | 0.761229 | 0.601655 | 0.309693 |
|  right(filter)  | 222.796692  | 0.767139 | 0.625296 | 0.332151 |
|  results(1-n):  |             |          |          |          |
|      left       |  27.406635  | 0.919431 | 0.826161 | 0.512986 |
|  left(filter)   |  27.141611  | 0.922654 | 0.84872  | 0.621422 |
|      right      | 941.315857  | 0.272038 |  0.1291  | 0.046825 |
|  right(filter)  | 234.795822  |  0.5291  | 0.347867 | 0.173081 |
|  results(n-1):  |             |          |          |          |
|      left       | 1104.678345 | 0.181692 | 0.092235 | 0.035644 |
|  left(filter)   |  330.43512  | 0.460942 | 0.30228  | 0.15519  |
|      right      |  29.978243  | 0.916792 | 0.830112 | 0.56961  |
|  right(filter)  |  29.770744  | 0.919338 | 0.845388 | 0.644486 |
|     results     |   (n-n):    |          |          |          |
|      left       | 140.606644  | 0.458214 | 0.211108 | 0.069218 |
|  left(filter)   |  45.662598  | 0.758356 | 0.574127 | 0.33131  |
|      right      | 102.383469  | 0.499763 | 0.235302 | 0.080909 |
|  right(filter)  |  33.339615  | 0.790765 | 0.602248 | 0.351757 |

| TransR results: |  Mean Rank  | Hits@10  |  Hits@3  |  Hits@1  |
| :-------------: | :---------: | :------: | :------: | :------: |
|      left       | 343.613678  | 0.368777 | 0.153138 | 0.022464 |
|  left(filter)   | 151.105743  | 0.593489 | 0.396506 | 0.119703 |
|      right      |  247.51358  | 0.417447 | 0.179902 | 0.02431  |
|  right(filter)  | 127.036766  | 0.634101 | 0.428772 | 0.125002 |
|     results     |             |          |          |          |
|      left       | 240.588333  | 0.425454 | 0.218144 | 0.093887 |
|  left(filter)   |  48.079632  | 0.68934  | 0.513433 | 0.301315 |
|      right      | 153.931976  | 0.494557 | 0.269946 | 0.130589 |
|  right(filter)  |  33.455166  | 0.73068  | 0.553029 | 0.329231 |
|  results(1-1):  |             |          |          |          |
|      left       | 172.769501  | 0.760047 | 0.576832 | 0.232861 |
|  left(filter)   |    172.5    | 0.764775 | 0.611111 | 0.232861 |
|      right      | 191.862885  | 0.747045 | 0.548463 | 0.232861 |
|  right(filter)  | 191.555557  | 0.751773 | 0.583924 | 0.234043 |
|  results(1-n):  |             |          |          |          |
|      left       | 160.893646  | 0.645498 | 0.426161 | 0.012701 |
|  left(filter)   | 160.598099  | 0.651564 | 0.461232 | 0.013649 |
|      right      | 1218.452271 | 0.17763  | 0.063697 | 0.003223 |
|  right(filter)  | 508.245117  | 0.337441 | 0.185024 | 0.011564 |
|  results(n-1):  |             |          |          |          |
|      left       | 1308.565308 | 0.122787 | 0.044092 | 0.00243  |
|  left(filter)   | 529.907532  | 0.281449 | 0.146048 | 0.010647 |
|      right      | 218.086441  | 0.624928 | 0.41604  | 0.013424 |
|  right(filter)  | 217.852097  | 0.629325 | 0.435713 | 0.015855 |
|  results(n-n):  |             |          |          |          |
|      left       | 179.114563  | 0.376334 | 0.13381  | 0.023517 |
|  left(filter)   |  74.361305  | 0.644158 | 0.433546 | 0.151437 |
|      right      | 137.391205  | 0.399242 | 0.140649 | 0.024961 |
|  right(filter)  |  61.378254  | 0.668104 | 0.453474 | 0.157711 |