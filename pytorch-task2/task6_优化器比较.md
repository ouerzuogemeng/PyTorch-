## 了解并比较不同的优化器
#### 总结每种优化器的优缺点，并附上代码示例，以便更好地理解
* SGD
* ASGD
* Rprop
* Adagrad
* Adadelta
* RMSprop
* Adam
* Adamax
* SparseAdam
* LBFGS

* BGD
* SGD
* MBGD
* Adagrad
* Adadelta
* RMSprop
* Adam
* Adamax
* SparseAdam
* LBFGS
#### 1. BDG(Batch gradient descent):
* 用整个训练集的数据来计算corss function的提督
   
   缺点：该方法在每次更新中，对整个数据集计算梯度，数据量较大时会导致计算非常缓慢；
        
	 无法新增样本
   
      代码示例：
      for i in range(nb_epochs):
          params_grad = evaluate_gradient(loss_function, data, params)
	      params = params - learning_rate * params_grad
		
#### 2. SGD(Stochastic gradient descent):
* 每次使用单个样本进行梯度更新

