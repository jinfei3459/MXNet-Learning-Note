####MxNet搭建CNN的主要步骤
1.	确定网络结构中的变量
2.	搭建网络结构，以及变量的处理方式
3.	通过SimpleBind函数构建网络
4.	进行前向计算，反向求导计算等

####搭建网络结构

```cpp
Symbol data = Symbol::Variable("data");
		Symbol data_label = Symbol::Variable("data_label");
		Symbol conv1_w = Symbol::Variable("conv1_w");
		Symbol conv1_b = Symbol::Variable("conv1_b");
        Symbol fc1_w = Symbol::Variable("fc1_w");
		Symbol fc1_b = Symbol::Variable("fc1_b");
		Symbol fc2_w = Symbol::Variable("fc2_w");
		Symbol fc2_b = Symbol::Variable("fc2_b");
        
		Symbol conv1 =
			Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 6);
		Symbol tanh1 = Activation("tanh1", conv1, ActivationActType::sigmoid);
		Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2),
			PoolingPoolType::avg, Shape(2, 2));
		Symbol flatten = Flatten("flatten", pool1);
		Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 2);
		Symbol tanh2 = Activation("tanh2", fc1, ActivationActType::sigmoid);
		Symbol fc2 = FullyConnected("fc2", tanh2, fc2_w, fc2_b, 2);
        
        args_map["data"] =
			NDArray(Shape(batch_size, 1, W, H), ctx_dev, false);
		NDArray::WaitAll();
		lenet.InferArgsMap(ctx_dev, &args_map, args_map);
        
        //省略将数据传入data和data_label的步骤
        
        Executor *exe = lenet.SimpleBind(ctx_dev, args_map);
		exe->Forward(true);
        exe->Backward();
		exe->UpdateAll(&opt, learning_rate, weight_decay);
		delete exe;

```
####确定网络结构中的变量
1.	定义前向通道使用的存储变量（输入值输出值和每层网络的参数等），统一压入`vector<mxnet::NDArray> in_args`
2.	定义反向计算通道的存储变量，统一压入`vector<mxnet::NDArray> arg_grad_store`
3.	每个变量的操作类型（write，add，noop），统一压入`vector<mxnet::OpReqType> grad_req_type`;
4.	定义aux_states（暂时不明白是什么作用，used as internal state in op），统一压入`vector<mxnet::NDArray>aux_states`

上述类型属于NDArray的可以只定义维数，不初始化值，维数类型可以根据InferShape（MXNet内部函数）得出，具体用法见cpp_net.hpp中的InitArgArrays函数

####Bind函数分析
`exe = mxnet::Executor::Bind(net, ctx_dev, g2c, in_args, arg_grad_store,
						grad_req_type, aux_states);`
![ind](.\pic\Bind.png)
-	symbol 网络的名称
-	default_ctx 网络默认的训练设备(CPU,GPU)
-	group2ctx  没有设置过，Context mapping group to context
-	in_args 输入值
-	arg_grad_store 梯度计算值
-	grad_req_type，每个梯度的保存方式,{kNullOp, kAddTo, kWriteTo}
-	aux_states ,  NDArray that is used as internal state in op

####执行训练过程
	exe->Forward(true);
	exe->Backward(std::vector<mxnet::NDArray>());
    optimizer->Update(i, &in_args[i], &arg_grad_store[i], learning_rate);