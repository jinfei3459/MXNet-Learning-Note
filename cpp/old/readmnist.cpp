#include "readmnist.h"
#include "opencv2/opencv.hpp"

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int read_Mnist(string filename, vector<float > &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					vec.push_back((float)temp/256.0);
				}
			}
			
		}
		return number_of_images;
	}
	
}



void read_Mnist_Label(string filename, vector<float> &vecl)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vecl.push_back( (float)temp);
		}
	}
}
int getplane(vector<float> &vec, vector<float> &vecl)
{
	char filename1[50], filename2[50], filename3[50], filename4[50];
	cv::Mat I;
	int i, number_of_images = 0;
	for (i = 1; i <= 500; i++)
	{
		sprintf(filename1, "F:\\zhubuntu\\plane\\1\\%d.bmp", i * 3 - 2);
		sprintf(filename3, "F:\\zhubuntu\\plane\\0\\%d.bmp", i * 3 - 2);
		I = cv::imread(filename1, 0);
		uchar* p = I.data;
		for (unsigned int j = 0; j < 4096; ++j)
		{
			vec.push_back(*p);
			p++;
		}
		vecl.push_back(1);
		number_of_images++;
		I = cv::imread(filename3, 0);
		p = I.data;
		for (unsigned int j = 0; j < 4096; ++j)
		{
			vec.push_back(*p);
			p++;
		}
		vecl.push_back(0);
		number_of_images++;
		sprintf(filename1, "F:\\zhubuntu\\plane\\1\\%d.bmp", i * 3 - 1);
		sprintf(filename3, "F:\\zhubuntu\\plane\\0\\%d.bmp", i * 3 - 1);
		I = cv::imread(filename1, 0);
		p = I.data;
		for (unsigned int j = 0; j < 4096; ++j)
		{
			vec.push_back(*p);
			p++;
		}
		vecl.push_back(1);
		number_of_images++;
		I = cv::imread(filename3, 0);
		p = I.data;
		for (unsigned int j = 0; j < 4096; ++j)
		{
			vec.push_back(*p);
			p++;
		}
		vecl.push_back(0);
		number_of_images++;
		sprintf(filename1, "F:\\zhubuntu\\plane\\1\\%d.bmp", i * 3);
		sprintf(filename3, "F:\\zhubuntu\\plane\\0\\%d.bmp", i * 3);
		I = cv::imread(filename1, 0);
		p = I.data;
		for (unsigned int j = 0; j < 4096; ++j)
		{
			vec.push_back(*p);
			p++;
		}
		vecl.push_back(1);
		number_of_images++;
		I = cv::imread(filename3, 0);
		p = I.data;
		for (unsigned int j = 0; j < 4096; ++j)
		{
			vec.push_back(*p);
			p++;
		}
		vecl.push_back(0);
		number_of_images++;
	}
	return number_of_images;
}
size_t Getdata(vector<float> &data, vector<float> &label)
{
	
	int number_of_images;
	
	vector<float > vec,vecl;

	/*read mnist*/
	string filename = "F:/zhubuntu/mxnet/mnist/train-images-idx3-ubyte";
	number_of_images=read_Mnist(filename, vec);
	filename = "F:/zhubuntu/mxnet/mnist/train-labels-idx1-ubyte";
	read_Mnist_Label(filename, vecl);
	
	/*read plane*/
	//number_of_images = getplane(vec, vecl);

	data.assign(vec.begin(), vec.end());
	label.assign(vecl.begin(), vecl.end());
	size_t N = number_of_images;
	return N;
}