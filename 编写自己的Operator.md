本文件根据http://mxnet.readthedocs.org/en/latest/developer-guide/operator.html 完成
可以和[卷积操作函数分析文档](./convolution-inl.h分析.md)一起看
#####操作接口
有Forward和Backward
```cpp
virtual void Forward(const OpContext &ctx,
                     const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_states) = 0;
```
```cpp
struct OpContext {
  int is_train;
  RunContext run_ctx;
  std::vector<Resource> requested;
}
```
```cpp
virtual void Backward(const OpContext &ctx,
                      const std::vector<TBlob> &out_grad,
                      const std::vector<TBlob> &in_data,
                      const std::vector<TBlob> &out_data,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &in_grad,
                      const std::vector<TBlob> &aux_states);
```
#####操作资源
-	Infershape
```cpp
virtual bool InferShape(std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        std::vector<TShape> *aux_shape) const = 0;
```
返回false当没有足够的输入来推断大小，返回error当数据大小不一致
-	所需资源
求解执行操作所需的前向和后向资源
```cpp
	virtual std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const;
virtual std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const;
```
其中ResourceRequest是一个表示所需资源的结构体

需要申请资源时，只需执行
```cpp
auto tmp_space_res = ctx.requested[kTempSpace].get_space(some_shape, some_stream);
auto rand_res = ctx.requested[kRandom].get_random(some_stream);
```
-	反向依赖
```cpp
virtual std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const;
```
利用此函数来声明反向传播所需要的参数，可以方便系统将不需要的内存释放掉
-	原址操作
当输入和输出大小相同时，通过声明表示输出可以覆盖输入的位置

#####生成操作
```cpp
\\OperatorProperty 中
virtual Operator* CreateOperator(Context ctx) const = 0;
```
例子
```cpp
class ConvolutionOp {
 public:
  void Forward( ... ) { ... }
  void Backward( ... ) { ... }
};
class ConvolutionOpProperty : public OperatorProperty {
 public:
  Operator* CreateOperator(Context ctx) const {
    return new ConvolutionOp;
  }
};
```
#####操作参数
首先定义一个ConvolutionParam结构体
```cpp
#include <dmlc/parameter.h>
struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel, stride, pad;
  uint32_t num_filter, num_group, workspace;
  bool no_bias;
};
```
将上述结构体放入ConvolutionOpProperty中并传递给operator类
```cpp
class ConvolutionOp {
 public:
  ConvolutionOp(ConvolutionParam p): param_(p) {}
  void Forward( ... ) { ... }
  void Backward( ... ) { ... }
 private:
  ConvolutionParam param_;
};
class ConvolutionOpProperty : public OperatorProperty {
 public:
  void Init(const vector<pair<string, string>& kwargs) {
    // initialize param_ using kwargs
  }
  Operator* CreateOperator(Context ctx) const {
    return new ConvolutionOp(param_);
  }
 private:
  ConvolutionParam param_;
};
```
#####在MXNet中注册操作
```cpp
DMLC_REGISTER_PARAMETER(ConvolutionParam);
MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionOpProperty);
```
其中第一个参数是名称，第二个是执行类
#####汇总
-	使用Operator 中的Forward和Backward写自己需要的操作
-	使用OperatorProperty的接口：
      -	将参数传递给操作类(可以使用Init接口)
      -	使用CreateOperator接口创造操作
      -	正确实现操作接口描述，例如参数名称
      -	正确实现InferShape设置输出张量大小
      -	[可选]如果需要其他资源，检查ForwardResource和BackwardResource
      -	[可选]如果Backward不需要Forward的所有输入和输出，检查DeclareBackwardDependency
      -	[可选]如果支持原址操作，检查ForwardInplaceOption和BackwardInplaceOption
-	在OperatorProperty中注册

####个人理解
MXNet关于operator的编写和使用主要可以分为以下几大部分，还是以conv为例
1.	operator的编写
	-	`struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam>`用来定义conv中用到的各种参数，但是不包括输入输出参数。
	-	`class ConvolutionOp : public Operator`主要实现前向和反向传播的定义
	-	`class ConvolutionProp : public OperatorProperty`主要实现外部接口，在`ConvolutionProp`会有一个，`Operator* CreateOperator(Context ctx) const override {}`调用`ConvolutionOp`，但是ConvolutionProp仍不和外部直接有接口
	-	所有以上完成之后需要注册
		-	`DMLC_REGISTER_PARAMETER(ConvolutionParam);`
		-	`MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
		.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
		.add_argument("weight", "Symbol", "Weight matrix.")
		.add_argument("bias", "Symbol", "Bias parameter.")
		.add_arguments(ConvolutionParam::__FIELDS__())
		.describe("Apply convolution to input then add a bias.");`可以看到，注册时接口为`ConvolutionProp`，其中`Convolution`为注册的名字，`add_argument`是为了方便使用者得到该操作的操作参数。
	其中`MXNET_REGISTER_OP_PROPERTY`的定义为
    ```cpp
   #define MXNET_REGISTER_OP_PROPERTY(name, OperatorPropertyType)          \
  DMLC_REGISTRY_REGISTER(::mxnet::OperatorPropertyReg, OperatorPropertyReg, name) \
  .set_body([]() { return new OperatorPropertyType(); })                \
  .set_return_type("Symbol") \
  .check_name()
  ```
2. 调用编写好的层次
	-	最直接的调用方法是
	```cpp
	OperatorProperty *OperatorProperty::Create(const char* type_name) {
  	auto *creator = dmlc::Registry<OperatorPropertyReg>::Find(type_name);//即在已经注册的层次中寻找type_name，如果寻找到就会返回一个OperatorProperty
  	if (creator == nullptr) {
    LOG(FATAL) << "Cannot find Operator " << type_name << " in registry";
 	 }
  return creator->body();
	}
	```
    但是该方法接口并没有直接在dll中对外开放
    -	现在使用的接口一般是
   	```cpp
   Symbol::Symbol(const std::string &operator_name, const std::string &name,
               std::vector<const char *> input_keys,
               std::vector<SymbolHandle> input_values,
               std::vector<const char *> config_keys,
               std::vector<const char *> config_values) {
	SymbolHandle handle;
	AtomicSymbolCreator creator = op_map_->GetSymbolCreator(operator_name);
	MXSymbolCreateAtomicSymbol(creator, config_keys.size(), config_keys.data(),
		config_values.data(), &handle);
	MXSymbolCompose(handle, operator_name.c_str(), input_keys.size(),
		input_keys.data(), input_values.data());
	blob_ptr_ = std::make_shared<SymBlob>(handle);
}
```
	-	其中`op_map_->GetSymbolCreator`定义如下

```cpp
class OpMap {
public:
  /*!
  * \brief Create an Mxnet instance
  */
  inline OpMap() {
    mx_uint num_symbol_creators = 0;
    AtomicSymbolCreator *symbol_creators = nullptr;
    int r =
      MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);
    CHECK_EQ(r, 0);
    for (mx_uint i = 0; i < num_symbol_creators; i++) {
      const char *name;
      const char *description;
      mx_uint num_args;
      const char **arg_names;
      const char **arg_type_infos;
      const char **arg_descriptions;
      const char *key_var_num_args;
      r = MXSymbolGetAtomicSymbolInfo(symbol_creators[i], &name, &description,
        &num_args, &arg_names, &arg_type_infos,
        &arg_descriptions, &key_var_num_args);
	 /*去除注释之后可以看到输出的是之前
     std::cout << name << i << std::endl;
	  if (i==17)
	  {
		  std::cout << name<<std::endl;
		  for (int j = 0; j < num_args;j++)
		  
		  std::cout << j<<*(arg_names+j)<<*(arg_descriptions+j)<<std::endl;
		 // std::cout << *arg_type_infos;
	  }*/
      CHECK_EQ(r, 0);
      symbol_creators_[name] = symbol_creators[i];
    }
  }

  /*!
  * \brief Get a symbol creator with its name.
  *
  * \param name name of the symbol creator
  * \return handle to the symbol creator
  */
  inline AtomicSymbolCreator GetSymbolCreator(const std::string &name) {
    return symbol_creators_[name];
  }
private:
  std::map<std::string, AtomicSymbolCreator> symbol_creators_;
};

```
####关于NDArray，symbol和operator的关系
一般来说，symbol中包含NDArray和operator，
node,表示每个symbol中的节点，node总共可以分为三类
-	正常node，包含一个图所要求的所有元素
-	操作，inputs_为空，表示一个未应用的操作
-	变量，sym_指向为空，表示一个张量

因此我们一般定义两种symbol，一种是纯数据的，例如输入的data和偏置权值等
第二种就是根据operator生成的symbol，在生成该symbol之时一般都会确定好输入权值等和第一类symbol的关系。
具体根据operator生成symbol可以看conv生成symbol的定义
```cpp
inline Symbol Convolution(const std::string& symbol_name,
                          Symbol data,
                          Symbol weight,
                          Symbol bias,
                          Shape kernel,
                          int num_filter,
                          Shape stride = Shape(1,1),
                          Shape dilate = Shape(1,1),
                          Shape pad = Shape(0,0),
                          int num_group = 1,
                          int64_t workspace = 512,
                          bool no_bias = false) {
  return Operator("Convolution")
           .SetParam("kernel", kernel)
           .SetParam("num_filter", num_filter)
           .SetParam("stride", stride)
           .SetParam("dilate", dilate)
           .SetParam("pad", pad)
           .SetParam("num_group", num_group)
           .SetParam("workspace", workspace)
           .SetParam("no_bias", no_bias)
           .SetInput("data", data)
           .SetInput("weight", weight)
           .SetInput("bias", bias)
           .CreateSymbol(symbol_name);
}
```