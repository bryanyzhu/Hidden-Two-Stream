#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/flow_transformer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FlowTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tFlow Transformer Layer:: LayerSetUp: \t";


	std::cout<<prefix<<"Getting output_H_ and output_W_"<<std::endl;
	output_H_ = bottom[0]->shape(2);
	output_W_ = bottom[0]->shape(3);
	std::cout<<prefix<<"output_H_ = "<<output_H_<<", output_W_ = "<<output_W_<<std::endl;

	std::cout<<prefix<<"Getting pre-defined parameters"<<std::endl;

	// check the validation for the parameter theta
	CHECK(bottom[1]->shape(2) == bottom[0]->shape(2)) << "The third dimension (height) of the flow field and " <<
			"U should be the same" << std::endl;
	CHECK(bottom[1]->shape(3) == bottom[0]->shape(3)) << "The forth dimension (width) of the flow field and " <<
			"U should be the same" << std::endl;

	// initialize the matrix for output grid
	std::cout<<prefix<<"Initializing the matrix for output grid"<<std::endl;

	//2 channel image, 1 channel each for U and V of flow field
	vector<int> shape_output(4);
	shape_output[0] = bottom[0]->shape(0);
	shape_output[1] = 2;
	shape_output[2] = output_H_;
	shape_output[3] = output_W_;
	output_grid.Reshape(shape_output);

	Dtype* data = output_grid.mutable_cpu_data();
	for(int n=0;n<bottom[0]->shape(0);n++){
	for(int r=0;r<output_H_;r++){
		for(int c=0;c<output_W_;c++){
			data[output_grid.offset(n,0,r,c)] = c;
			data[output_grid.offset(n,1,r,c)] = r;
		}
	}
	}

	// initialize the matrix for input grid
	std::cout<<prefix<<"Initializing the matrix for input grid"<<std::endl;

	vector<int> shape_input(4);
	shape_input[0] = bottom[0]->shape(0);
	shape_input[1] = 2;
	shape_input[2] = output_H_;
	shape_input[3] = output_W_;
	input_grid.Reshape(shape_input);

	std::cout<<prefix<<"Initialization finished."<<std::endl;
}

template <typename Dtype>
void FlowTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	string prefix = "\t\tFlow Transformer Layer:: Reshape: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = output_H_;
	shape[3] = output_W_;

	top[0]->Reshape(shape);

	// reshape dTheta_tmp
	vector<int> dTheta_tmp_shape(4);

	dTheta_tmp_shape[0] = N;
	dTheta_tmp_shape[1] = 2;
	dTheta_tmp_shape[2] = 3;
	dTheta_tmp_shape[3] = output_H_ * output_W_ * C;

	dTheta_tmp.Reshape(dTheta_tmp_shape);

	// init all_ones_2
	vector<int> all_ones_2_shape(1);
	all_ones_2_shape[0] = output_H_ * output_W_ * C;
	all_ones_2.Reshape(all_ones_2_shape);

	// reshape full_theta
	vector<int> full_theta_shape(2);
	full_theta_shape[0] = N;
	full_theta_shape[1] = 6;
	full_theta.Reshape(full_theta_shape);

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
Dtype FlowTransformerLayer<Dtype>::transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py) {
	/*
	so i think this does the bilinear interpolation for a single pixel
	*/

	bool debug = false;

	string prefix = "\t\tFlow Transformer Layer:: transform_forward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!\t"<<std::endl;
	if(debug) std::cout<<prefix<<"(px, py) = ("<<px<<", "<<py<<")"<<std::endl;

	Dtype res = (Dtype)0.;

	Dtype x = px; Dtype y = py;

	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"1: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"2: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"3: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"4: (m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));
		res += w * pic[m * W + n];
		if(debug) std::cout<<prefix<<"w = "<<w<<", pic[m, n] = "<<pic[m * W + n]<<std::endl;
	}

	if(debug) std::cout<<prefix<<"Finished. \tres = "<<res<<std::endl;

	return res;
}

template <typename Dtype>
void FlowTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tFlow Transformer Layer:: Forward_cpu: \t";

	/*
	CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
			" CHECK in st_layer.cpp file. Line number: 240-241." << std::endl;
	*/

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	const Dtype* U = bottom[0]->cpu_data();
	const Dtype* flows = bottom[1]->cpu_data();
	
	const Dtype* output_grid_data = output_grid.cpu_data();

	Dtype* input_grid_data = input_grid.mutable_cpu_data();
	Dtype* V = top[0]->mutable_cpu_data();

	caffe_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_set(top[0]->count(), (Dtype)0, V);

	//combine flow with grid
	caffe_add(input_grid.count(),output_grid_data,flows,input_grid_data);

	// for each input
	for(int i = 0; i < N; ++i) {
		//coordinates is a matrix of coordinates on the original image
		//to sample pixel values from
		//Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		//Dtype* coordinates = input_grid_data + input_grid.offset(i,0,0,0);

		//do affine transformation
		//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, 3, (Dtype)1.,
		//      output_grid_data, theta + 6 * i, (Dtype)0., coordinates);
		
		Dtype px, py;

		//for each pixel on the output, find the interpolated pixel value
		for(int s = 0; s < output_H_; ++s){
			for(int t = 0; t < output_W_; ++t) {
				//px,py should be the sample coordinates on the source img

				py = input_grid_data[input_grid.offset(i,0,s,t)];
				px = input_grid_data[input_grid.offset(i,1,s,t)];

				for(int j = 0; j < C; ++j){
					//do interpolation
					V[top[0]->offset(i, j, s, t)] = transform_forward_cpu(
							U + bottom[0]->offset(i, j, 0, 0), px, py);
				}
			}
		}
	}

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void FlowTransformerLayer<Dtype>::transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
		const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy) {
	/*
	U is the input image
	V is out output image, dV is local gradient from output
	*/

	bool debug = false;

	string prefix = "\t\tFlow Transformer Layer:: transform_backward_cpu: \t";

	if(debug) std::cout<<prefix<<"Starting!"<<std::endl;

	//Dtype x = (px + 1) / 2 * H; Dtype y = (py + 1) / 2 * W;
	Dtype x = px; Dtype y = py;
	if(debug) std::cout<<prefix<<"(x, y) = ("<<x<<", "<<y<<")"<<std::endl;

	int m, n; Dtype w;

	m = floor(x); n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		if(to_compute_dU_) dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x) + 1; n = floor(y); w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		if(to_compute_dU_) dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x); n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		if(to_compute_dU_) dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	if(debug) std::cout<<prefix<<"(m, n) = ("<<m<<", "<<n<<")"<<std::endl;

	if(m >= 0 && m < H && n >= 0 && n < W) {
		w = max(0, 1 - abs(x - m)) * max(0, 1 - abs(y - n));

		if(to_compute_dU_) dU[m * W + n] += w * dV;

		if(abs(x - m) < 1) {
			if(m >= x) {
				dpx += max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx += "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			} else {
				dpx -= max(0, 1 - abs(y - n)) * U[m * W + n] * dV;// * H / 2;
				if(debug) std::cout<<prefix<<"dpx -= "<<max(0, 1 - abs(y - n))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<H / 2<<std::endl;
			}
		}

		if(abs(y - n) < 1) {
			if(n >= y) {
				dpy += max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy += "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			} else {
				dpy -= max(0, 1 - abs(x - m)) * U[m * W + n] * dV;// * W / 2;
				if(debug) std::cout<<prefix<<"dpy -= "<<max(0, 1 - abs(x - m))<<" * "<<U[m * W + n]<<" * "<<dV<<" * "<<W / 2<<std::endl;
			}
		}
	}

	if(debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void FlowTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		string prefix = "\t\tFlow Transformer Layer:: Backward_cpu: \t";

		//CHECK(false) << "Don't use the CPU implementation! If you really want to, delete the" <<
		//		" CHECK in st_layer.cpp file. Line number: 420-421." << std::endl;

		if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

		const Dtype* dV = top[0]->cpu_diff();
		Dtype* input_grid_data = input_grid.mutable_cpu_data();
		//const Dtype* output_grid_data = output_grid.cpu_data(); // remove if you don't need to recalc the input grid
		const Dtype* U = bottom[0]->cpu_data();

		Dtype* dU = bottom[0]->mutable_cpu_diff(); //we won't be setting this
		//Dtype* dTheta = bottom[1]->mutable_cpu_diff();
		//const Dtype* flows = bottom[1]->cpu_data(); // remove if you don't need to recalc the input grid
		Dtype* flow_diff = bottom[1]->mutable_cpu_diff();

		caffe_set(bottom[0]->count(), (Dtype)0, dU);
		//caffe_set(bottom[1]->count(), (Dtype)0, dTheta);
		caffe_set(input_grid.count(), (Dtype)0, flow_diff);

		//calc input grid data, recalc incase blob is reset between forward and backward
		//caffe_add(input_grid.count(),output_grid_data,flows,input_grid_data);

		for(int i = 0; i < N; ++i) {

			//const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
			//Dtype* coordinates_diff = input_grid_diff + (output_H_ * output_W_ * 2) * i;

			const Dtype* curInputGrid = &input_grid_data[input_grid.offset(i,0,0,0)];
			//since its just element wise addition, we are
			//just going to set the diff directly to the flow
			Dtype* curFlowDiff = &flow_diff[bottom[1]->offset(i,0,0,0)];

			//int row_idx;
			Dtype px, py, delta_dpx, delta_dpy;

			for(int s = 0; s < output_H_; ++s)
				for(int t = 0; t < output_W_; ++t) {

					//row_idx = output_W_ * s + t;	//this means fix!

					px = curInputGrid[input_grid.offset(0,1,s,t)];
					py = curInputGrid[input_grid.offset(0,0,s,t)];

					for(int j = 0; j < C; ++j) {

						delta_dpx = delta_dpy = (Dtype)0.;

						transform_backward_cpu(dV[top[0]->offset(i, j, s, t)], U + bottom[0]->offset(i, j, 0, 0),
								px, py, dU + bottom[0]->offset(i, j, 0, 0), delta_dpx, delta_dpy);

						curFlowDiff[bottom[1]->offset(0,1,s,t)] += delta_dpx;
						curFlowDiff[bottom[1]->offset(0,0,s,t)] += delta_dpy;
					}

					/*
					dpx = curFlowDiff[row_idx * 2];
					dpy = curFlowDiff[row_idx * 2 + 1];

					dTheta[6 * i] += dpx * (s * 1.0 / output_H_ * 2 - 1);
					dTheta[6 * i + 1] += dpx * (t * 1.0 / output_W_ * 2 - 1);
					dTheta[6 * i + 2] += dpx;
					dTheta[6 * i + 3] += dpy * (s * 1.0 / output_H_ * 2 - 1);
					dTheta[6 * i + 4] += dpy * (t * 1.0 / output_W_ * 2 - 1);
					dTheta[6 * i + 5] += dpy;
					*/
				}
		}

		if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(FlowTransformerLayer);
#endif

INSTANTIATE_CLASS(FlowTransformerLayer);
REGISTER_LAYER_CLASS(FlowTransformer);

}  // namespace caffe
