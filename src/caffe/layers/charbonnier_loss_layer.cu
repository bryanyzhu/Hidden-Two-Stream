#include <vector>

#include "caffe/layers/charbonnier_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CharbonnierLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//blob 0 the difference image

	int eleCount = bottom[0]->count();
	Dtype alpha = this->layer_param_.charbonnier_loss_param().alpha(); //robustness
	Dtype beta = this->layer_param_.charbonnier_loss_param().beta(); //pre-scaling
	Dtype epsilonSq = this->layer_param_.charbonnier_loss_param().epsilon();
	epsilonSq = epsilonSq*epsilonSq;

	//robust loss for photometric
	//sum( ((x*b)^2 + epsilon^2 )^alpha )
	//x*b
	caffe_gpu_scale(eleCount,
		beta,
		bottom[0]->gpu_data(),
		diff_.mutable_gpu_data());
	//square, is there no caffe_sqr() for gpu?
	caffe_gpu_powx(eleCount,
		diff_.mutable_gpu_data(),
		(Dtype)2,
		diff_.mutable_gpu_data());
	//+ epsilon^2
	caffe_gpu_add_scalar(eleCount,
		epsilonSq,
		diff_.mutable_gpu_data());
	//^alpha
	caffe_gpu_powx(eleCount,
		diff_.mutable_gpu_data(),
		alpha,
		diff_.mutable_gpu_data());
	//use mask if exists
	if(bottom.size() == 2){
		caffe_gpu_mul(eleCount,
			bottom[1]->gpu_data(),
			diff_.mutable_gpu_data(),
			diff_.mutable_gpu_data());
	}
	//sum
	Dtype photometricLoss;
	caffe_gpu_asum(eleCount,diff_.gpu_data(),&photometricLoss);

	//output and average
	int batchPixels = 0;
	if(bottom.size() == 2){ // if there is a mask finf # of valid pixels
		Dtype validMaskPixels;
		caffe_gpu_asum(bottom[1]->count(), bottom[1]->gpu_data(), &validMaskPixels);
		batchPixels = bottom[0]->shape(0) * validMaskPixels;
	}
	else{
		batchPixels = bottom[0]->shape(0) * bottom[0]->shape(2)*bottom[0]->shape(3);
	}

	top[0]->mutable_cpu_data()[0] = photometricLoss/batchPixels;
}

template <typename Dtype>
void CharbonnierLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//there should only be one top blob
	//there should only be one bottom blob

	//int N = top[0]->shape(0);
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	//calculate local derivative: 2*x*b*b*alpha*((x*b)^2 + epsilon^2)^(alpha-1)
	int eleCount = bottom[0]->count();
	Dtype alpha = this->layer_param_.charbonnier_loss_param().alpha(); //robustness
	Dtype beta = this->layer_param_.charbonnier_loss_param().beta(); //pre-scaling
	Dtype betaSq = beta*beta;
	Dtype epsilonSq = this->layer_param_.charbonnier_loss_param().epsilon();
	epsilonSq = epsilonSq*epsilonSq;

	//x*b
	caffe_gpu_scale(eleCount,
		beta,
		bottom[0]->gpu_data(),
		diff_.mutable_gpu_diff());
	//square
	caffe_gpu_powx(eleCount,
		diff_.mutable_gpu_diff(),
		(Dtype)2,
		diff_.mutable_gpu_diff());
	//+ epsilon^2
	caffe_gpu_add_scalar(eleCount,
		epsilonSq,
		diff_.mutable_gpu_diff());
	//^alpha
	caffe_gpu_powx(eleCount,
		diff_.mutable_gpu_diff(),
		alpha - 1,
		diff_.mutable_gpu_diff());
	//*x
	caffe_gpu_mul(eleCount,
		bottom[0]->gpu_data(),
		diff_.gpu_diff(),
		bottom_diff);

	//combine with global derivative, *2*alpha*b*b, and average over examples
	int batchPixels = 0;
	if(bottom.size() == 2){ // if there is a mask finf # of valid pixels
		Dtype validMaskPixels;
		caffe_gpu_asum(bottom[1]->count(), bottom[1]->gpu_data(), &validMaskPixels);
		batchPixels = bottom[0]->shape(0) * validMaskPixels;
	}
	else{
		batchPixels = bottom[0]->shape(0) * bottom[0]->shape(2)*bottom[0]->shape(3);
	}
	caffe_gpu_scal(eleCount,
		top_diff[0]*2*alpha*betaSq/batchPixels,
		bottom_diff);

	//use mask if exists
	if(bottom.size() == 2){
		caffe_gpu_mul(eleCount,
			bottom[1]->gpu_data(),
			bottom_diff,
			bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(CharbonnierLossLayer);

}  // namespace caffe
