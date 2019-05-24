## 了解并比较不同的优化器
#### 总结每种优化器的优缺点，并附上代码示例，以便更好地理解

## 常用的3种优化器
* BGD
* SGD
> class torch.optim.SGD(params, lr= 0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
* MBGD
### 其他优化算法
* Momentum
> 超参数
* NAG
> 超参数
* Adagrad
> class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial _accumulator_value=0)
* Adadelta 
> class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial _accumulator_value=0)
* Rmsprop
> class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e- 08, weight_decay=0, momentum=0, centered=False)
* Adam
> class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e- 08, weight_decay=0, amsgrad=False)
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
								
「挑战1」
	
   2. 此方法是对所有参数更新时应用同样的学习率，如果我们想对一些特定的特征（如部分出现频率低的特征）进行大一点的更新，则难以做到
   
「挑战2」
   
   3. 对于非凸函数，该方法容易被困在局部极小值，或者鞍点处。（鞍点周围的error都是一样的，所有维度的梯度都接近于0）
   
「挑战3」
   
   ![](./photo/saddle_point)
   
	代码示例：
      for i in range(nb_epochs):
      	np.random.shuffle(data):
		for batch in get_batches(data,batch_size=50):  #每次随机抽取50个样本计算梯度 n取值一般在50-256之间
			params_grad = evaluate_gradient(loss_function, batch, params)
			params = params - learning_rate * params_grad

## 为了解决优化器存在的上述3点挑战，引入了下面几种算法：
### 针对挑战1:
#### 1.1 Momentum(动量)
[参考资料](https://www.cnblogs.com/jungel24/p/5682612.html)

在 ravines（即曲面的一个方向比另一个方向更陡） 的情况下，由于梯度方向一般朝着更陡的方向，SGD会发生震荡而迟迟不能接近极小值（学习率越大，震荡越严重）。
在这种情况下，引入Monmentum来抑制震荡（即使得小球更倾向于往x方向行进--往前走，而不是上下摆动）

当我们将一个小球从山上滚下来时，没有阻力的话，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。 

加入的这一项，可以使得梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。

该超参数一般设定为0.9
	θ=𝛾 * 𝜈t-1 +  𝜂* 𝛻θ* J(θ)
	θ = θ - 𝜈t
	(t,t-1,𝛻θ中θ为下标)

缺点：这种情况相当于小球盲目地沿着坡滚，而不具备先知（如快要上坡时，如果知道需要减速了的话，适应性会更好）

为了解决这个问题，引入了Nesterov算法。

#### 1.2 Nesterov accelerated gradient

	θ=𝛾 * 𝜈t-1 +  𝜂* 𝛻θ* J(θ-𝛾 * 𝜈t-1)
	θ = θ - 𝜈t
	(t,t-1,𝛻θ中θ为下标)
用θ-𝛾 * 𝜈t-1来近似当作参数下一步会变成的值，则在计算梯度时，不是在当前位置，而是在未来位置上

比较:
* Momentum计算当前的梯度，然后在更新后的累积梯度后会有一个大的跳跃
* NAG会先在前一步的累积梯度上有一个较大的跳跃，然后衡量一下梯度做一下修正，这种预期的更新可以避免我们走的太快。

通过以上方式，我们可以做到，在更新梯度时顺应loss function的梯度来调整速度，并且对SGD进行加速。

该超参数一般设定为0.9

### 针对挑战2---希望可以根据参数的重要性而对不同的参数进行不同程度的更新
#### 2.1 Adagrad
该算法可以对低频的参数做较大的更新，对高频的做较小的更新。

参数𝜂一般取0.01

* 优点：减少了学习率的手动调节

因此，该算法对于稀疏的数据表现很好（需要在低频的特征上有较大的更新），提高了SGD的鲁棒性。

* 缺点：分母会不断积累，学习率会收缩并最终会变得非常小(该算法将𝛻θ* J(θ)除以过去的梯度平方和)

#### 2.2 Adadelta
该算法是对Adagrad的改进。将分母换成了过去的梯度平方的衰减平均值，相当于梯度的平方根（RMS）。

此外，还将学习率𝜂换成了RMS[∆θ]，我们甚至不需要提前设定学习率了。

该超参数一般设定为0.9

#### 2.3 RMSprop
RMSprop 和 Adadelta都是为了解决Adagrad学习率急剧下降的问题。

RMSprop是一种自适应学习率方法。

公式跟Adadelta类似（学习率设置方式不同）

Hinton 建议设定 γ 为 0.9, 学习率 η 为 0.001。

#### Adam
另一种计算每个参数的自适应学习率的方法

即像Adadelta 和 RMSprop一样储存了过去梯度平方的指数衰减平均值，也像momentum一样保持了过去梯度的指数衰减平方值。

超参数设定值: 

建议 β1 ＝ 0.9，β2 ＝ 0.999，ϵ ＝ 10e−8

实践表明，Adam 比其他适应性学习方法效果要好。

## 如何选择优化器
   很多论文里都会用SGD，没有momentum等，虽然能达到极小值，但是比其他算法用的时间长，而且可能会被困在鞍点。
   
   如果数据是稀疏的，就用自适应方法。即 Adagrad, Adadelta, RMSprop, Adam
   * RMSprop, Adadelta, Adam 在很多情况下的效果是相似的。
   * Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum
   
   整体来讲，Adam 是最好的选择。
   
   
   
## 其他方法：
* torch.optim.ASGD 随机平均梯度下降
> class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000. 0, weight_decay=0)
* torch.optim.Rprop
> class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
>> 该方法适合full_batch,不适用于mini_batch,因而很少见到。
* torch.optim.Adamax
> class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e- 08, weight_decay=0)
>> 该方法是对Adam增加了一个学习率上限的概念。
* torch.optim.SparseAdam
> class torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
>> 针对稀疏张量的一种“阉割版”Adam优化方法
* torch.optim.LBFGS
> class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_ grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
>> L-BFGS 属于拟牛顿算法。L-BFGS 是对 BFGS 的改进，特点就是节省内存。

