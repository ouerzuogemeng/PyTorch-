task3 未加入激活函数



从自身理解的角度进行说明：

分类问题激活函数一般使用sigmoid/softmax，其中sigmoid解决二分类问题，转化之后标签0 和1 的概率之和为1；
softmax解决多分类问题，转化之后多个类别概率之和为1。

如果未使用softmax转化，也可输出最大值对应的标签，只是各个标签对应的预测值之和不等于1。

故虽然未加入激活函数，但最终结果是不变的。

如果需要加入激活函数，可借鉴task2。


正常逻辑回归写法：


    class LogisticRegression(nn.Module):
        def __init__(self,input_size,num_classes):
            super(LogisticRegression,self).__init__()
            self.linear = nn.Linear(input_size,num_classes)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.linear(x)
            out = self.sigmoid(out)
            return out
