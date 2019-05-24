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
   
   缺点：1）该方法在每次更新中，对整个数据集计算梯度，数据量较大时会导致计算非常缓慢；
        
	2）无法新增样本
   
      代码示例：
      for i in range(nb_epochs):
          params_grad = evaluate_gradient(loss_function, data, params)
	      params = params - learning_rate * params_grad
		
#### 2. SGD(Stochastic gradient descent):
* 每次使用单个样本进行梯度更新
优点：每次只更新一个样本，训练速度较快，且可以新增样本

缺点：更新较频繁，会造成cross function有严重震荡

	代码示例：
      for i in range(nb_epochs):
      	np.random.shuffle(data):
	for sample in data:  #每次随机抽取一个样本计算梯度
		params_grad = evaluate_gradient(loss_function, sample, params)
		params = params - learning_rate * params_grad

#### 3. MBGD(Mini-batch gradient descent):
* 每次使用一个批次的样本进行梯度更新
优点：前两种方法的优化，既可以加快训练速度，又可以降低参数更新时的方差，使收敛更稳定

缺点：1. 与learning_rate的大小有关。如果学习率过小，收敛速度会很慢；如果过大，则会在极小值处不停地震荡甚至偏离；
     2. 

	代码示例：
      for i in range(nb_epochs):
      	np.random.shuffle(data):
	for batch in get_batches(data,batch_size=50):  #每次随机抽取50个样本计算梯度 n取值一般在50-256之间
		params_grad = evaluate_gradient(loss_function, batch, params)
		params = params - learning_rate * params_grad
