### 使用电脑版本:mac
### 踩坑记录：
> 1. tensorboardX pip安装失败。
>> 原因可能是电脑上安装了不同版本的python，有冲突。
>> 解决方法：通过git安装 
>>> git clone https://github.com/lanpa/tensorboardX && cd tensorboardX && python setup.py install
>>>> 安装成功
> 2. 运行 tensorboard --logdir=<../pytorch/runs>  具体路径略去
>> 报错：Failed to load the native TensorFlow runtime
>>> 解决方法：重新安装tensorflow (使用pip报错，改为conda安装)
>>>> conda install tensorflow
>>>>> 成功！！


### 在jupter中运行好可视化代码之后，打开命令行输入 tensorboard --logdir=<../pytorch/runs> ,显示：
> TensorBoard 1.13.1 at http://...local:6006 (Press CTRL+C to quit)   #注：具体地址略去
>> 可以浏览器中贴入网址愉快滴玩耍啦。。
