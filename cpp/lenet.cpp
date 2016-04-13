#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <stdint.h>
#include <vector>
#include "MxNetCpp.h"
#include "readmnist.h"
using namespace std;
using namespace mxnet::cpp;

class Lenet {
public:
	Lenet()
		: ctx_cpu(Context(DeviceType::kCPU, 0)),
		ctx_dev(Context(DeviceType::kGPU, 0)) {}
	void Run() {
		/*
		* LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
		* "Gradient-based learning applied to document recognition."
		* Proceedings of the IEEE (1998)
		* */

		/*define the symbolic net*/
		

		for (auto s : lenet.ListArguments()) {
			LG << s;
		}

		/*setup basic configs*/
		int val_fold = 3;
	
		int batch_size = 20;
		int max_epoch = 50;
		float learning_rate = 4e-6;
		float weight_decay = 1e-5;

		/*prepare the data*/
		vector<float> data_vec, label_vec;
		//size_t data_count = GetData(&data_vec, &label_vec);
		size_t data_count = Getdata(data_vec, label_vec);
		const float *dptr = data_vec.data();
		const float *lptr = label_vec.data();
		NDArray data_array = NDArray(Shape(data_count, 1, W, H), ctx_cpu,
			false);  // store in main memory, and copy to
		// device memory while training
		NDArray label_array =
			NDArray(Shape(data_count), ctx_cpu,
			false);  // it's also ok if just store them all in device memory
		data_array.SyncCopyFromCPU(dptr, data_count * W * H);
		label_array.SyncCopyFromCPU(lptr, data_count);
		data_array.WaitToRead();
		label_array.WaitToRead();

		size_t train_num = data_count * (1 - val_fold / 10.0);
		train_data = data_array.Slice(0, train_num);
		train_label = label_array.Slice(0, train_num);
		val_data = data_array.Slice(train_num, data_count);
		val_label = label_array.Slice(train_num, data_count);

		LG << "here read fin";

		/*init some of the args*/
		// map<string, NDArray> args_map;
		args_map["data"] =
			NDArray(Shape(batch_size, 1, W, H), ctx_dev, false);

		args_map["data"] = data_array.Slice(0, batch_size).Copy(ctx_dev);
		//args_map["data_label"] = label_array.Slice(0, batch_size).Copy(ctx_dev);
		/*args_map["fc1_weight"] =
			NDArray(Shape(500, 4 * 4 * 50), ctx_dev, false);
		//SampleGaussian(0, 1, &args_map["fc1_weight"]);
		args_map["fc1_weight"] = 0.3;
		args_map["fc2_bias"] = NDArray(Shape(10), ctx_dev, false);
		args_map["fc2_bias"] = 0;*/
		NDArray::WaitAll();

		LG << "here slice fin";
		/*
		* we can also feed in some of the args other than the input all by
		* ourselves,
		* fc2-w , fc1-b for example:
		* */
		// args_map["fc2_w"] =
		// NDArray(mshadow::Shape2(500, 4 * 4 * 50), ctx_dev, false);
		// NDArray::SampleGaussian(0, 1, &args_map["fc2_w"]);
		// args_map["fc1_b"] = NDArray(mshadow::Shape1(10), ctx_dev, false);
		// args_map["fc1_b"] = 0;

		Optimizer opt("ccsgd", learning_rate, weight_decay);
		opt.SetParam("momentum", 0.9)
			.SetParam("rescale_grad", 1.0)
			.SetParam("clip_gradient", 10);

		for (int ITER = 0; ITER < max_epoch; ++ITER) {
			size_t start_index = 0;
			while (start_index < train_num) {
				if (start_index + batch_size > train_num) {
					start_index = train_num - batch_size;
				}
				args_map["data"] =
					train_data.Slice(start_index, start_index + batch_size)
					.Copy(ctx_dev);
				args_map["data_label"] =
					train_label.Slice(start_index, start_index + batch_size)
					.Copy(ctx_dev);
				start_index += batch_size;
				NDArray::WaitAll();

				Executor *exe = lenet.SimpleBind(ctx_dev, args_map);
				exe->Forward(true);
				exe->Backward();
				exe->UpdateAll(&opt, learning_rate, weight_decay);

				delete exe;
			}

			LG << "Iter " << ITER
				<< ", accuracy: " << ValAccuracy(batch_size , lenet);
		}
	}
	void Model()
	{
		Symbol data = Symbol::Variable("data");
		Symbol data_label = Symbol::Variable("data_label");
		Symbol conv1_w = Symbol::Variable("conv1_w");
		Symbol conv1_b = Symbol::Variable("conv1_b");
		Symbol conv2_w = Symbol::Variable("conv2_w");
		Symbol conv2_b = Symbol::Variable("conv2_b");
		Symbol conv3_w = Symbol::Variable("conv3_w");
		Symbol conv3_b = Symbol::Variable("conv3_b");
		Symbol conv4_w = Symbol::Variable("conv4_w");
		Symbol conv4_b = Symbol::Variable("conv4_b");
		Symbol fc1_w = Symbol::Variable("fc1_w");
		Symbol fc1_b = Symbol::Variable("fc1_b");
		Symbol fc2_w = Symbol::Variable("fc2_w");
		Symbol fc2_b = Symbol::Variable("fc2_b");

		Symbol conv1 =
			Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 6);
		Symbol tanh1 = Activation("tanh1", conv1, ActivationActType::sigmoid);
		Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2),
			PoolingPoolType::avg, Shape(2, 2));

		Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b,
			Shape(5, 5), 12);
		Symbol tanh2 = Activation("tanh2", conv2, ActivationActType::sigmoid);
		Symbol pool2 = Pooling("pool2", tanh2, Shape(2, 2),
			PoolingPoolType::avg, Shape(2, 2));

		Symbol conv3 = Convolution("conv3", pool2, conv3_w, conv3_b,
			Shape(5, 5), 24);
		Symbol drop3 = Dropout("drop3", conv3, 0.1);
		Symbol tanh3 = Activation("tanh3", drop3, ActivationActType::sigmoid);
		
		Symbol pool3 = Pooling("pool3", tanh3, Shape(2, 2),
			PoolingPoolType::avg, Shape(2, 2));

		Symbol conv4 = Convolution("conv3", pool3, conv4_w, conv4_b,
			Shape(5, 5), 48);
		Symbol tanh4 = Activation("tanh3", conv4, ActivationActType::sigmoid);
		Symbol pool4 = Pooling("pool3", tanh4, Shape(2, 2),
			PoolingPoolType::avg, Shape(2, 2));

		Symbol flatten = Flatten("flatten", pool4);
		Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 3);
		//Symbol tanh4 = Activation("tanh4", fc1, ActivationActType::sigmoid);
		//Symbol fc2 = FullyConnected("fc2", tanh4, fc2_w, fc2_b, 10);

		lenet = SoftmaxOutput("softmax", fc1, data_label);
		args_map["data"] =
			NDArray(Shape(1, 1, W, H), ctx_dev, false);
		lenet.InferArgsMap(ctx_dev, &args_map, args_map);
	}
	void Save()
	{

		int n = 1;
		int i = 1;
		for (auto s : lenet.ListArguments()) {
			if ((s == "data") || (s == "data_label"))
			{
				i = i + 1;
				continue;
			}
			char filename[50];
			sprintf(filename, "%d.data", i);
			vector<mx_uint> shape_out = args_map[s].GetShape();
			for (int j = 0; j < shape_out.size(); j++)
			{
				n = n*shape_out[j];
			}
			float *dptr = new float[n];
			NDArray save_cpu = args_map[s].Copy(ctx_cpu);
			NDArray::WaitAll();

			save_cpu.SyncCopyToCPU(dptr, n);
			NDArray::WaitAll();
	

			ofstream out(filename, ios::binary);

			out.write((char*)dptr, n*sizeof(float));


			out.close();
			delete[] dptr;
			n = 1;
			i++;
		}
	}
	void Test(vector<float > &vec, int data_count)
	{
		args_map["data"] =
			NDArray(Shape(data_count, 1, W, H), ctx_dev, false);
		args_map["data_label"] =
			NDArray(Shape(data_count), ctx_dev, false);
		const float *dptr = vec.data();
		args_map["data"].SyncCopyFromCPU(dptr, data_count * W * H);;

		NDArray::WaitAll();


		Executor *exe = lenet.SimpleBind(ctx_dev, args_map);
		exe->Forward(false);

		const auto &out = exe->outputs;
		NDArray out_cpu = out[0].Copy(ctx_cpu);

		NDArray::WaitAll();


		const mx_float *dptr_out = out_cpu.GetData();
		//cout << out_cpu.GetShape()[0];

		for (int i = 0; i < data_count; ++i) {

			int cat_num = out_cpu.GetShape()[1];
			float p_label = 0, max_p = dptr_out[i * cat_num];
			for (int j = 0; j < cat_num; ++j) {
				float p = dptr_out[i * cat_num + j];

				if (max_p < p) {
					p_label = j;
					max_p = p;
				}

			}
			cout << p_label << endl;

		}



		delete exe;
	}
	void Load()
	{
		

		int n = 1;
		int i = 1;
		for (auto s : lenet.ListArguments()) {
			if ((s == "data") || (s == "data_label"))
			{
				i = i + 1;
				continue;
			}
			char filename[50];
			sprintf(filename, "%d.data", i);
			vector<mx_uint> shape_out = args_map[s].GetShape();
			for (int j = 0; j < shape_out.size(); j++)
			{
				n = n*shape_out[j];
			}
			vector<float> dp;
			ifstream in(filename, ios::binary);
			for (int j = 0; j < n; j++)
			{
				float s;
				in.read((char*)&s, sizeof(s));
				dp.push_back(s);

			}

			float *dptr = dp.data();

			args_map[s].SyncCopyFromCPU(dptr, n);
			NDArray::WaitAll();

			in.close();
			n = 1;
			i++;
			//cout << endl;
		}
		

	}
private:
	Context ctx_cpu;
	Context ctx_dev;
	map<string, NDArray> args_map;
	NDArray train_data;
	NDArray train_label;
	NDArray val_data;
	NDArray val_label;
	Symbol lenet;
	int W = 200;
	int H = 200;

	size_t GetData(vector<float> *data, vector<float> *label) {
		const char *train_data_path = "./train.csv";
		ifstream inf(train_data_path);
		string line;
		inf >> line;  // ignore the header
		size_t _N = 0;
		while (inf >> line) {
			for (auto &c : line) c = (c == ',') ? ' ' : c;
			stringstream ss;
			ss << line;
			float _data;
			ss >> _data;
			label->push_back(_data);
			while (ss >> _data) data->push_back(_data / 256.0);
			_N++;
		}
		inf.close();
		return _N;
	}

	float ValAccuracy(int batch_size, Symbol lenet) {
		size_t val_num = val_data.GetShape()[0];

		size_t correct_count = 0;
		size_t all_count = 0;

		size_t start_index = 0;
		while (start_index < val_num) {
			if (start_index + batch_size > val_num) {
				start_index = val_num - batch_size;
			}
			args_map["data"] =
				val_data.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
			args_map["data_label"] =
				val_label.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
			start_index += batch_size;
			NDArray::WaitAll();

			Executor *exe = lenet.SimpleBind(ctx_dev, args_map);
			exe->Forward(false);

			const auto &out = exe->outputs;
			NDArray out_cpu = out[0].Copy(ctx_cpu);
			NDArray label_cpu =
				val_label.Slice(start_index - batch_size, start_index).Copy(ctx_cpu);

			NDArray::WaitAll();

			const mx_float *dptr_out = out_cpu.GetData();
			const mx_float *dptr_label = label_cpu.GetData();
			for (int i = 0; i < batch_size; ++i) {
				float label = dptr_label[i];
				int cat_num = out_cpu.GetShape()[1];
				float p_label = 0, max_p = dptr_out[i * cat_num];
				for (int j = 0; j < cat_num; ++j) {
					float p = dptr_out[i * cat_num + j];
					if (max_p < p) {
						p_label = j;
						max_p = p;
					}
				}
				if (p_label == 1) p_label=38;
				if (p_label == 0) p_label =49;
				if (label == p_label) correct_count++;
				//cout << p_label << endl;
			}
			all_count += batch_size;

			delete exe;
		}
		return correct_count * 1.0 / all_count;
	}
};

int main(int argc, char const *argv[]) {
	Lenet lenet;
	lenet.Model();
	lenet.Load();
	lenet.Run();
	lenet.Save();
	//
	//lenet.Run();
	return 0;
}