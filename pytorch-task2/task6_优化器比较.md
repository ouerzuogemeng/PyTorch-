## 了解并比较不同的优化器
#### 总结每种优化器的优缺点，并附上代码示例，以便更好地理解

## 常用的3种优化器
* BGD
* SGD
* MBGD
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
     （有一种解决方案是先设置大一点的学习率，当两次迭代之间的变化低于某个阈值后，就减小学习率，但这个阈值要根据数据集的特点提前写好）
   
   2. 此方法是对所有参数更新时应用同样的学习率，如果我们想对一些特定的特征（如部分出现频率低的特征）进行大一点的更新，则难以做到
   
   3. 对于非凸函数，该方法容易被困在局部极小值，或者鞍点处。（鞍点周围的error都是一样的，所有维度的梯度都接近于0）
   ![](./photo/saddle_point)
   
	代码示例：
      for i in range(nb_epochs):
      	np.random.shuffle(data):
	for batch in get_batches(data,batch_size=50):  #每次随机抽取50个样本计算梯度 n取值一般在50-256之间
		params_grad = evaluate_gradient(loss_function, batch, params)
		params = params - learning_rate * params_grad

## 为了解决优化器存在的上述3点挑战，引入了下面几种算法：
#### 1.Momentum(动量)
[参考资料](https://www.cnblogs.com/jungel24/p/5682612.html)
个人总结：

在 ravines（即曲面的一个方向比另一个方向更陡） 的情况下，由于梯度方向一般朝着更陡的方向，SGD会发生震荡而迟迟不能接近极小值（学习率越大，震荡越严重）。
在这种情况下，引入Monmentum来抑制震荡（即使得小球更倾向于往x方向行进--往前走，而不是上下摆动）

当我们将一个小球从山上滚下来时，没有阻力的话，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。 

加入的这一项，可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。

	θ=𝛾 * 𝜈t-1 +  𝜂* 𝛻θ* J(θ)
	θ = θ - 𝜈t
	(t,t-1,𝛻θ中θ为下标)

缺点：这种情况相当于小球盲目地沿着坡滚，而不具备先知（如快要上坡时，如果知道需要减速了的话，适应性会更好）

为了解决这个问题，引入了Nesterov算法。

#### 5.Nesterov accelerated gradient

	θ=𝛾 * 𝜈t-1 +  𝜂* 𝛻θ* J(θ-𝛾 * 𝜈t-1)
	θ = θ - 𝜈t
	(t,t-1,𝛻θ中θ为下标)
用θ-𝛾 * 𝜈t-1来近似当作参数下一步会变成的值，则在计算梯度时，不是在当前位置，而是在未来位置上

比较:
* Momentum计算当前的梯度，然后在更新后的累积梯度后会有一个大的跳跃
* NAG会先在
