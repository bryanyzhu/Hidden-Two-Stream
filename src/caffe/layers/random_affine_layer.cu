#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/random_affine_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void createTransforms(const int nthreads, 
	Dtype* tx, Dtype* ty, Dtype* rot, Dtype* scal, Dtype* flip,
	Dtype* t1, Dtype* t2) {

	CUDA_KERNEL_LOOP(n, nthreads) {
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
	}
}

template <typename Dtype>
void RandomAffineLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//read in transformation distributions
	Dtype max_translation_x = this->layer_param_.random_affine_param().max_translation_x();
	Dtype max_translation_y = this->layer_param_.random_affine_param().max_translation_y();
	Dtype max_rotation = this->layer_param_.random_affine_param().max_rotation();
	Dtype min_scale = this->layer_param_.random_affine_param().min_scale();
	Dtype max_scale = this->layer_param_.random_affine_param().max_scale();
	bool horizontalFlipping = this->layer_param_.random_affine_param().horizontal_flipping();

	//helpers
	Dtype* t1 = t1_.mutable_gpu_data();
	Dtype* t2 = t2_.mutable_gpu_data();
	Dtype* t3 = t3_.mutable_gpu_data();
	Dtype* bottomExpandedData = bottomExpanded.mutable_gpu_data();
	Dtype* t4 = t4_.mutable_gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	caffe_gpu_set(t1_.count(), (Dtype)0, t1);
	caffe_gpu_set(t2_.count(), (Dtype)0, t2);
	caffe_gpu_set(t3_.count(), (Dtype)0, t3);

	int N = top[0]->shape(0);

	//generate transform params
	Dtype* tx = translation_x.mutable_gpu_data();
	Dtype* ty = translation_y.mutable_gpu_data();
	Dtype* rot = rotation.mutable_gpu_data();
	Dtype* scal = scale.mutable_gpu_data();
	Dtype* flip = flipping_rand.mutable_gpu_data();

	caffe_gpu_rng_uniform(N,-max_translation_x,max_translation_x,tx);
	caffe_gpu_rng_uniform(N,-max_translation_y,max_translation_y,ty);
	caffe_gpu_rng_uniform(N,-max_rotation,max_rotation,rot);
	caffe_gpu_rng_uniform(N,min_scale,max_scale,scal);
	if(horizontalFlipping){
		caffe_gpu_rng_uniform(N,(Dtype)0,(Dtype)1,flip);
	}
	else{
		caffe_gpu_set(N, (Dtype)0, flip);
	}

	//create transforms
	createTransforms<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, 
				tx,ty,rot,scal,flip,t1,t2);

	//copy transforms
	for(int n=0;n<N;n++){
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 3, 3, 3, (Dtype)1.0,
		      &t1[9*n], &t2[9*n], (Dtype)0.0, &t3[9*n]);

		if(bottom.size() == 1){
			caffe_copy(6, &(bottom[0]->mutable_gpu_data())[6*n], &bottomExpandedData[9*n]);
			bottomExpanded.mutable_cpu_data()[9*n + 8] = 1;

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 3, 3, 3, (Dtype)1.0,
		      		&t3[9*n], &bottomExpandedData[9*n], (Dtype)0.0, &t4[9*n]);
			//caffe_copy(6, &t4[9*n], &top_data[6*n]);
			caffe_copy(6, &bottomExpandedData[9*n], &top_data[6*n]);
		}
		else{
			caffe_copy(6, &t3[9*n], &top_data[6*n]);
		}
	}
}

template <typename Dtype>
void RandomAffineLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top) {
	//do nothing
}


INSTANTIATE_LAYER_GPU_FUNCS(RandomAffineLayer);

}	// namespace caffe
