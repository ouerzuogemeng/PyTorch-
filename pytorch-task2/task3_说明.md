task3 未加入激活函数



从自身理解的角度进行说明：

分类问题激活函数一般使用sigmoid/softmax，其中sigmoid解决二分类问题，转化之后标签0 和1 的概率之和为1；
softmax解决多分类问题，转化之后多个类别概率之和为1。

模型中使用的CrossEntropyLoss()损失函数，是经过softmax激活函数之后，再计算其与traget的交叉熵损失，
即该方法是nn.LogSolfrmax()（激活函数）和nn.NLLLoss()（严格意义上的交叉熵函数）进行了结合

也可以在训练model中直接加入softmax激活函数，可借鉴task2。
