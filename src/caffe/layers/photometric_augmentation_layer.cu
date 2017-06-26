#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/photometric_augmentation_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

template <typename Dtype>
void PhotometricAugmentationLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
	//do nothing
}

template <typename Dtype>
__global__ void transform(const int nthreads, const int pixelCount, const int channelPixelCount, const int channels,
	Dtype* noise, Dtype* contrast, Dtype* brightness, Dtype* colour, Dtype* gamma,
	const Dtype* bottom, Dtype* top) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / pixelCount;
		int c = (index / channelPixelCount) % channels;

		const Dtype curBottom = bottom[index];
		Dtype* curTop = &top[index];

		//a*I + b
		Dtype pixel = curBottom*(contrast[n] + 1) + brightness[n];

		//color
		pixel = pixel*colour[3*n+c];

		//clamp values
		if(pixel < 0){
			pixel = 0;
		}
		else if(pixel > 1){
			pixel = 1;
		}

		//gamma
		Dtype gammaInv = 1/gamma[n];
		*curTop = pow(pixel, gammaInv);		
	}
}

template <typename Dtype>
__global__ void subtractMean(const int nthreads, const int channelPixelCount, const int channels,
	const Dtype* means, Dtype* top) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		int c = (index / channelPixelCount) % channels;
		top[index] = top[index] - means[c];
	}
}

template <typename Dtype>
void PhotometricAugmentationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
	Dtype* contrastData = contrast.mutable_gpu_data();
	Dtype* brightnessData = brightness.mutable_gpu_data();
	Dtype* colourData = colour.mutable_gpu_data();
	Dtype* gammaData = gamma.mutable_gpu_data();
	const Dtype* meanData = means.gpu_data();

	int N = top[0]->shape(0);
	int pixelCount = top[0]->shape(1)*top[0]->shape(2)*top[0]->shape(3);
	int channelPixelCount = top[0]->shape(2)*top[0]->shape(3);
	int channels = top[0]->shape(1);

	//generate paramaters
	if (noise_sigma>0){
		for(int i=0;i<bottom.size();i++){
			caffe_gpu_rng_gaussian(noise[i]->count(), (Dtype)0, noise_sigma, noise[i]->mutable_gpu_data());
		}
	}
	else{
		for(int i=0;i<bottom.size();i++){
			caffe_set(noise[i]->count(), (Dtype)0, noise[i]->mutable_gpu_data());
		}
	}
	caffe_gpu_rng_uniform(N,contrast_min,contrast_max,contrastData);
	if(brightness_sigma > 0){
		caffe_gpu_rng_gaussian(N, (Dtype)0, brightness_sigma, brightnessData);
	}
	else{
		caffe_set(N, (Dtype)0, brightnessData);
	}
	caffe_gpu_rng_uniform(N*3,colour_min,colour_max,colourData);
	caffe_gpu_rng_uniform(N,gamma_min,gamma_max,gammaData);

	for(int b=0;b<bottom.size();b++){
		const Dtype* bottom_data = bottom[b]->gpu_data();
		Dtype* topData = top[b]->mutable_gpu_data();
		Dtype* noiseData = noise[b]->mutable_gpu_data();

		//4 other transforms
		int nthreads = top[0]->count();
		transform<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, pixelCount, channelPixelCount, channels,
					noiseData, contrastData,brightnessData,colourData,gammaData,
					bottom_data, topData);

		//combine noise
		caffe_gpu_axpy(top[0]->count(), (Dtype)1, noiseData, topData);

		//mean subtraction
		subtractMean<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, channelPixelCount, channels,
			meanData, topData);
	}
}


INSTANTIATE_LAYER_GPU_FUNCS(PhotometricAugmentationLayer);

}	// namespace caffe
