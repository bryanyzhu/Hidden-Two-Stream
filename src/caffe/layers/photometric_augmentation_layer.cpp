#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/photometric_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PhotometricAugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//allocate noise blob
	for(int i=0;i<bottom.size();i++){
		noise.push_back(new Blob<Dtype>);
	}
}

template <typename Dtype>
PhotometricAugmentationLayer<Dtype>::~PhotometricAugmentationLayer(){
	for(int i=0;i<noise.size();i++){
		delete noise[i];
	}
}

template <typename Dtype>
void PhotometricAugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//check bottoms are same size
	for(int i=1;i<bottom.size();i++){
		CHECK_EQ(bottom[i]->shape(0), bottom[0]->shape(0)) << "All bottoms must have the same dimensions";
		CHECK_EQ(bottom[i]->shape(1), bottom[0]->shape(1)) << "All bottoms must have the same dimensions";
		CHECK_EQ(bottom[i]->shape(2), bottom[0]->shape(2)) << "All bottoms must have the same dimensions";
		CHECK_EQ(bottom[i]->shape(3), bottom[0]->shape(3)) << "All bottoms must have the same dimensions";
	}

	//reshape tops
	for(int i=0;i<bottom.size();i++){
		top[i]->ReshapeLike(*bottom[0]);
	}
	int batchSize = bottom[0]->shape(0);
	int channels = bottom[0]->shape(1);

	//check mean subtraction params are valid
	int num_mean_params = this->layer_param_.photometric_augmentation_param().mean_subtraction_size();
	if (num_mean_params > 0){
		//check if means == channels

		CHECK_EQ(channels, num_mean_params) << "There must be a mean_subtraction param for every input channel";

		hasMeans = true;
		vector<int> shape_mean(1);
		shape_mean[0] = num_mean_params;
		shape_mean[1] = 1;
		shape_mean[2] = 1;
		shape_mean[3] = 1;
		means.Reshape(shape_mean);

		//copy data to means
		Dtype* meansData = means.mutable_cpu_data();
		for(int i=0;i<num_mean_params;i++){
			meansData[i] = (Dtype)this->layer_param_.photometric_augmentation_param().mean_subtraction(i);
		}
	}
	else{
		hasMeans = false;
	}

	//reshape helper blobs
	vector<int> shape_t_params(4);
	shape_t_params[0] = batchSize;
	shape_t_params[1] = 1;
	shape_t_params[2] = 1;
	shape_t_params[3] = 1;

	vector<int> shape_t_params2(4);
	shape_t_params2[0] = batchSize;
	shape_t_params2[1] = 1;
	shape_t_params2[2] = 1;
	shape_t_params2[3] = 3;

	for(int i=0;i<bottom.size();i++){
		noise[i]->ReshapeLike(*bottom[0]);
	}
	contrast.Reshape(shape_t_params);
	brightness.Reshape(shape_t_params);
	colour.Reshape(shape_t_params2);
	gamma.Reshape(shape_t_params);
}

template <typename Dtype>
void PhotometricAugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//get parameters
	Dtype contrast_min = this->layer_param_.photometric_augmentation_param().contrast_min();
	Dtype contrast_max = this->layer_param_.photometric_augmentation_param().contrast_max();
	Dtype brightness_sigma = this->layer_param_.photometric_augmentation_param().brightness_sigma();
	Dtype colour_min = this->layer_param_.photometric_augmentation_param().colour_min();
	Dtype colour_max = this->layer_param_.photometric_augmentation_param().colour_max();
	Dtype gamma_min = this->layer_param_.photometric_augmentation_param().gamma_min();
	Dtype gamma_max = this->layer_param_.photometric_augmentation_param().gamma_max();
	Dtype noise_sigma = this->layer_param_.photometric_augmentation_param().noise_sigma();

	//Dtype mean_subtraction = this->layer_param_.photometric_augmentation_param().mean_subtraction();

	//helpers
	Dtype* contrastData = contrast.mutable_cpu_data();
	Dtype* brightnessData = brightness.mutable_cpu_data();
	Dtype* colourData = colour.mutable_cpu_data();
	Dtype* gammaData = gamma.mutable_cpu_data();
	const Dtype* meanData = means.cpu_data();

	int N = top[0]->shape(0);

	//generate paramaters
	if (noise_sigma>0){
		for(int i=0;i<bottom.size();i++){
			caffe_rng_gaussian(noise[i]->count(), (Dtype)0, noise_sigma, noise[i]->mutable_cpu_data());
		}
	}
	else{
		for(int i=0;i<bottom.size();i++){
			caffe_set(noise[i]->count(), (Dtype)0, noise[i]->mutable_cpu_data());
		}
	}
	caffe_rng_uniform(N,contrast_min,contrast_max,contrastData);
	if(brightness_sigma > 0){
		caffe_rng_gaussian(N, (Dtype)0, brightness_sigma, brightnessData);
	}
	else{
		caffe_set(N, (Dtype)0, brightnessData);
	}
	caffe_rng_uniform(N*3,colour_min,colour_max,colourData);
	caffe_rng_uniform(N,gamma_min,gamma_max,gammaData);

	//transform
	for(int b=0;b<bottom.size();b++){
		Dtype* noiseData = noise[b]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[b]->cpu_data();
		Dtype* topData = top[b]->mutable_cpu_data();

		for(int n=0;n<N;n++){
			const Dtype* curBottom = &bottom_data[bottom[b]->offset(n,0,0,0)];
			Dtype* curTop = &topData[top[b]->offset(n,0,0,0)];
			Dtype* curNoise =  &noiseData[noise[b]->offset(n,0,0,0)];
			int channelPixels = bottom[b]->shape(2) * bottom[b]->shape(3);
			int pixels = bottom[b]->shape(1) * channelPixels;

			//a*I + b
			caffe_axpy(pixels,contrastData[n] + 1,curBottom,curTop);
			caffe_add_scalar(pixels, brightnessData[n], curTop);

			//color
			caffe_scal(channelPixels,colourData[3*n],curTop);
			caffe_scal(channelPixels,colourData[3*n+1],&curTop[channelPixels]);
			caffe_scal(channelPixels,colourData[3*n+2],&curTop[channelPixels*2]);

			//clamp values
			for(int i=0;i<pixels;i++){
				if(curTop[i] < 0){
					curTop[i]  = 0;
				}
				else if(curTop[i] > 1){
					curTop[i] = 1;
				}
			}

			//gamma
			Dtype gammaInv = 1/gammaData[n];
			caffe_powx(pixels,curTop,gammaInv,curTop);

			//combine noise
			caffe_axpy(pixels, (Dtype)1, curNoise, curTop);

			//subtract mean
			caffe_add_scalar(channelPixels,-meanData[0],curTop);
			caffe_add_scalar(channelPixels,-meanData[1],&curTop[channelPixels]);
			caffe_add_scalar(channelPixels,-meanData[2],&curTop[channelPixels*2]);
		}
	}
}

template <typename Dtype>
void PhotometricAugmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
	//do nothing
}



#ifdef CPU_ONLY
STUB_GPU(PhotometricAugmentationLayer);
#endif

INSTANTIATE_CLASS(PhotometricAugmentationLayer);
REGISTER_LAYER_CLASS(PhotometricAugmentation);

}  // namespace caffe
