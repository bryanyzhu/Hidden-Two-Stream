#include <vector>

#include "caffe/layers/charbonnier_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CharbonnierLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// LossLayers have a non-zero (1) loss by default.
	if (this->layer_param_.loss_weight_size() == 0) {
		this->layer_param_.add_loss_weight(Dtype(1));
	}
}


template <typename Dtype>
void CharbonnierLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	if(bottom.size() == 2){ //if there is a mask, check size
		CHECK_EQ(bottom[0]->shape(0),bottom[1]->shape(0)) << "Mask batch size not the same as input batch size";
		CHECK_EQ(bottom[0]->shape(2),bottom[1]->shape(2)) << "Mask must be the same size as input";
		CHECK_EQ(bottom[0]->shape(3),bottom[1]->shape(3)) << "Mask must be the same size as input";
	}

	vector<int> shape(4);
	shape[0] = 1;
	shape[1] = 1;
	shape[2] = 1;
	shape[3] = 1;
	top[0]->Reshape(shape);

	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CharbonnierLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
	caffe_cpu_scale(eleCount,
		beta,
		bottom[0]->cpu_data(),
		diff_.mutable_cpu_data());
	//square
	caffe_sqr(eleCount,
		diff_.mutable_cpu_data(),
		diff_.mutable_cpu_data());
	//+ epsilon^2
	caffe_add_scalar(eleCount,
		epsilonSq,
		diff_.mutable_cpu_data());
	//^alpha
	caffe_powx(eleCount,
		diff_.mutable_cpu_data(),
		alpha,
		diff_.mutable_cpu_data());
	//use mask if exists
	if(bottom.size() == 2){
		caffe_mul(eleCount,
			bottom[1]->cpu_data(),
			diff_.mutable_cpu_data(),
			diff_.mutable_cpu_data());
	}

	//sum
	Dtype photometricLoss = caffe_cpu_asum(eleCount,diff_.cpu_data());

	//output and average
	int batchPixels = 0;
	if(bottom.size() == 2){ // if there is a mask finf # of valid pixels
		batchPixels = bottom[0]->shape(0) * caffe_cpu_asum(bottom[1]->count(), bottom[1]->cpu_data());
	}
	else{
		batchPixels = bottom[0]->shape(0) * bottom[0]->shape(2)*bottom[0]->shape(3);
	}
	top[0]->mutable_cpu_data()[0] = photometricLoss/batchPixels;
}

template <typename Dtype>
void CharbonnierLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//there should only be one top blob
	//there should only be one bottom blob

	//int N = top[0]->shape(0);
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	//calculate local derivative: 2*x*b*b*alpha*((x*b)^2 + epsilon^2)^(alpha-1)
	int eleCount = bottom[0]->count();
	Dtype alpha = this->layer_param_.charbonnier_loss_param().alpha(); //robustness
	Dtype beta = this->layer_param_.charbonnier_loss_param().beta(); //pre-scaling
	Dtype betaSq = beta*beta;
	Dtype epsilonSq = this->layer_param_.charbonnier_loss_param().epsilon();
	epsilonSq = epsilonSq*epsilonSq;

	//x*b
	caffe_cpu_scale(eleCount,
		beta,
		bottom[0]->cpu_data(),
		diff_.mutable_cpu_diff());
	//square
	caffe_sqr(eleCount,
		diff_.cpu_diff(),
		diff_.mutable_cpu_diff());
	//+ epsilon^2
	caffe_add_scalar(eleCount,
		epsilonSq,
		diff_.mutable_cpu_diff());
	//^(alpha-1)
	caffe_powx(eleCount,
		diff_.mutable_cpu_diff(),
		alpha - 1,
		diff_.mutable_cpu_diff());
	//*x
	caffe_mul(eleCount,
		bottom[0]->cpu_data(),
		diff_.cpu_diff(),
		bottom_diff);

	//combine with global derivative, *2*alpha*b*b, and average over examples
	int batchPixels = 0;
	if(bottom.size() == 2){ // if there is a mask finf # of valid pixels
		batchPixels = bottom[0]->shape(0) * caffe_cpu_asum(bottom[1]->count(), bottom[1]->cpu_data());
	}
	else{
		batchPixels = bottom[0]->shape(0) * bottom[0]->shape(2)*bottom[0]->shape(3);
	}
	caffe_scal(eleCount,
		top_diff[0]*2*alpha*betaSq/batchPixels,
		bottom_diff);

	//use mask if exists
	if(bottom.size() == 2){
		caffe_mul(eleCount,
			bottom[1]->cpu_data(),
			bottom_diff,
			bottom_diff);
	}
}

#ifdef CPU_ONLY
STUB_GPU(CharbonnierLayer);
#endif

INSTANTIATE_CLASS(CharbonnierLossLayer);
REGISTER_LAYER_CLASS(CharbonnierLoss);

}  // namespace caffe
