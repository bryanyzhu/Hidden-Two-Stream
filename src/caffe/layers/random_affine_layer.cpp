#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/random_affine_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {


template <typename Dtype>
void RandomAffineLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//do nothing
}

template <typename Dtype>
void RandomAffineLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//set in batch size
	uint32_t batchSize = this->layer_param_.random_affine_param().batch_size();
	vector<int> shape_top(4);
	shape_top[0] = batchSize;
	shape_top[1] = 1;
	shape_top[2] = 2;
	shape_top[3] = 3;
	top[0]->Reshape(shape_top);

	//check valid bottom
	if(bottom.size() == 1){
		CHECK_EQ(top[0]->shape(0), bottom[0]->shape(0)) << "Bottom batch size not the same";
		CHECK_EQ(top[0]->shape(1), bottom[0]->shape(1)) << "Bottom should have 1 channel";
		CHECK_EQ(top[0]->shape(2), bottom[0]->shape(2)) << "Bottom not 2x3 affine";
		CHECK_EQ(top[0]->shape(3), bottom[0]->shape(3)) << "Bottom not 2x3 affine";
	}

	//set working variables
	vector<int> shape_transform(4);
	shape_transform[0] = batchSize;
	shape_transform[1] = 1;
	shape_transform[2] = 3;
	shape_transform[3] = 3;

	t1_.Reshape(shape_transform);
	t2_.Reshape(shape_transform);
	t3_.Reshape(shape_transform);
	bottomExpanded.Reshape(shape_transform);
	t4_.Reshape(shape_transform);

	//transform params
	vector<int> shape_param(4);
	shape_param[0] = batchSize;
	shape_param[1] = 1;
	shape_param[2] = 1;
	shape_param[3] = 1;

	translation_x.Reshape(shape_param);
	translation_y.Reshape(shape_param);
	rotation.Reshape(shape_param);
	scale.Reshape(shape_param);
	flipping_rand.Reshape(shape_param);
}

template <typename Dtype>
void RandomAffineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//read in transformation distributions
	Dtype max_translation_x = this->layer_param_.random_affine_param().max_translation_x();
	Dtype max_translation_y = this->layer_param_.random_affine_param().max_translation_y();
	Dtype max_rotation = this->layer_param_.random_affine_param().max_rotation();
	Dtype min_scale = this->layer_param_.random_affine_param().min_scale();
	Dtype max_scale = this->layer_param_.random_affine_param().max_scale();
	bool horizontalFlipping = this->layer_param_.random_affine_param().horizontal_flipping();

	//helpers
	Dtype* t1 = t1_.mutable_cpu_data();
	Dtype* t2 = t2_.mutable_cpu_data();
	Dtype* t3 = t3_.mutable_cpu_data();
	Dtype* bottomExpandedData = bottomExpanded.mutable_cpu_data();
	Dtype* t4 = t4_.mutable_cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	caffe_set(t1_.count(), (Dtype)0, t1);
	caffe_set(t2_.count(), (Dtype)0, t2);
	caffe_set(t3_.count(), (Dtype)0, t3);
	caffe_set(bottomExpanded.count(), (Dtype)0, bottomExpandedData);
	caffe_set(t4_.count(), (Dtype)0, t4);

	int N = top[0]->shape(0);

	//generate transform params
	Dtype* tx = translation_x.mutable_cpu_data();
	Dtype* ty = translation_y.mutable_cpu_data();
	Dtype* rot = rotation.mutable_cpu_data();
	Dtype* scal = scale.mutable_cpu_data();
	Dtype* flip = flipping_rand.mutable_cpu_data();

	caffe_rng_uniform(N,-max_translation_x,max_translation_x,tx);
	caffe_rng_uniform(N,-max_translation_y,max_translation_y,ty);
	caffe_rng_uniform(N,-max_rotation,max_rotation,rot);
	caffe_rng_uniform(N,min_scale,max_scale,scal);
	if(horizontalFlipping){
		caffe_rng_uniform(N,(Dtype)0,(Dtype)1,flip);
	}
	else{
		caffe_set(N, (Dtype)0, flip);
	}

	for(int n=0;n<N;n++){
		//translation
		t1[9*n + 2] = tx[n];
		t1[9*n + 5] = ty[n];
		t1[9*n + 8] = 1;
		//rotation
		Dtype ang = rot[n];
		t1[9*n] = cos(ang); t1[9*n + 1] = -sin(ang);
		t1[9*n + 3] = sin(ang); t1[9*n + 4] = cos(ang);

		//scale
		Dtype scaleFactor = scal[n];
		scaleFactor = 1/scaleFactor;
		if(flip[n] > 0.5){
			t2[9*n] = scaleFactor;
			t2[9*n + 4] = -scaleFactor;
			t2[9*n + 8] = 1;
		}
		else{
			t2[9*n] = scaleFactor;
			t2[9*n + 4] = scaleFactor;
			t2[9*n + 8] = 1;
		}

		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 3, 3, 3, (Dtype)1.0,
		      &t1[9*n], &t2[9*n], (Dtype)0.0, &t3[9*n]);

		//transfer to top
		if(bottom.size() == 1){
			caffe_copy(6, &(bottom[0]->mutable_cpu_data()[6*n]), &bottomExpandedData[9*n]);
			bottomExpandedData[9*n + 8] = 1;

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 3, 3, 3, (Dtype)1.0,
		      		&t3[9*n], &bottomExpandedData[9*n], (Dtype)0.0, &t4[9*n]);
			caffe_copy(6, &t4[9*n], &top_data[6*n]);
		}
		else{
			caffe_copy(6, &t3[9*n], &top_data[6*n]);
		}
	}
}

template <typename Dtype>
void RandomAffineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
	//do nothing
}


#ifdef CPU_ONLY
STUB_GPU(RandomAffineLayer);
#endif

INSTANTIATE_CLASS(RandomAffineLayer);
REGISTER_LAYER_CLASS(RandomAffine);

}  // namespace caffe
