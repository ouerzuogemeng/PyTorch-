# PyTorch-
PyTorch入门
1. 为什么要学习PyTorch?

   PyTorch为深度学习框架的一种，目前不同的机构使用的深度学习框架可能不同，多了解一种学习框架有利于自己学习更好的学习深度学习。
   
   PyTorch的优势：
   
   1）支持GPU；
   
   2）动态神经网络；
   
      通过一种反向自动求导技术，可以让你零延迟地任意改变神经网络的行为。尽管这项技术不是PyTorch所独有，但它是目前为止实现最快的，也是对比Tensorflow最大的优势。
      
   3）python优先
   
      PyTorch使用了python作为开发语言
      
   4）命令式语言
   
      设计思路线性、直观、易于使用，没有异步的世界观
      
   5）轻松扩展
   
      源代码更加简洁直观，容易看懂
      
      
2. PyTorch的安装

   最初搜了安装教程，是直接pip安装到虚拟环境，安装到一半心血来潮自行退出，改为通过官网安装，结果安装成功之后，执行import torch报错，查询报错原因，意思文件存在冲突。
   
   最终解决方式是把anaconda卸载重装之后，又按照一下步骤重新安装了pytorch
   
   安装步骤：
   1）已安装anaconda
   
   2）点击进入pytorch官网(https://pytorch.org)
   
      选择自己的系统、版本等信息，会生成对应的命令行代码。
      
      复制Run this Command:框中的代码，在命令行中执行，即可安装成功。
      
      
      本人mac、python3.7在conda环境下安装，执行代码为：conda install pytorch torchvision -c pytorch


3. PyTorch入门

   由于现在自己也处于对PyTorch的初步了解和入门阶段，故通过对相关资料的依葫芦画瓢，再加上自身的理解进行总结。
   
   PyTorch基础主要需要了解一下模块：
   
   PyTorch 张量、数学运算、自动求导模块、最优化模块和神经网络模块
      
      
4. PyTorch入门学习教程 & 构建一个简单的深度学习网络

   入门代码(https://github.com/ouerzuogemeng/PyTorch-/blob/master/pytorch入门.ipynb)
