#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// #include "caffe/data_layers.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/rgb_flow_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe{
template <typename Dtype>
RgbFlowDataLayer<Dtype>:: ~RgbFlowDataLayer<Dtype>(){
	this->StopInternalThread();
	// this->JoinPrefetchThread();
}

template <typename Dtype>
void RgbFlowDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int new_height  = this->layer_param_.rgb_flow_data_param().new_height();
	const int new_width  = this->layer_param_.rgb_flow_data_param().new_width();
	const int new_length  = this->layer_param_.rgb_flow_data_param().new_length();
	const int num_segments = this->layer_param_.rgb_flow_data_param().num_segments();
	const string& source = this->layer_param_.rgb_flow_data_param().source();

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	vector<string> filenames(2);
	int label;
	int length;
	while (infile >> filenames[0] >> filenames[1] >> length >> label){
		lines_.push_back(std::make_pair(filenames,label));
		lines_duration_.push_back(length);
	}
	if (this->layer_param_.rgb_flow_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideoFlows();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	Datum datum;
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_]/num_segments;
	vector<int> offsets;
	for (int i = 0; i < num_segments; ++i){
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset = (*frame_rng)() % (average_duration - new_length + 1);
		offsets.push_back(offset+i*average_duration);
	}

	// Read both RGB frames and flow, always have new_length consecutive frames and new_length-1 optical flow between the frames
	CHECK(ReadSegmentRGBFlowToDatum(lines_[lines_id_].first[0], lines_[lines_id_].first[1], lines_[lines_id_].second,
									 offsets, new_height, new_width, new_length, &datum));
	
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.rgb_flow_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		    this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
		}
		// this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		    this->prefetch_[i].data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		}
		// this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    	this->prefetch_[i].label_.Reshape(batch_size, 1, 1, 1);
  	}
	// this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void RgbFlowDataLayer<Dtype>::ShuffleVideoFlows(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void RgbFlowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch){

	Datum datum;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());
	Dtype* top_data = batch->data_.mutable_cpu_data();
	Dtype* top_label = batch->label_.mutable_cpu_data();
	RgbFlowDataParameter rgb_flow_data_param = this->layer_param_.rgb_flow_data_param();
	const int batch_size = rgb_flow_data_param.batch_size();
	const int new_height = rgb_flow_data_param.new_height();
	const int new_width = rgb_flow_data_param.new_width();
	const int new_length = rgb_flow_data_param.new_length();
	const int num_segments = rgb_flow_data_param.num_segments();
	const int lines_size = lines_.size();

	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
		for (int i = 0; i < num_segments; ++i){
			if (this->phase_==TRAIN){
				if (average_duration >= new_length){
					caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
					int offset = (*frame_rng)() % (average_duration - new_length + 1);
					offsets.push_back(offset+i*average_duration);
				} else {
					offsets.push_back(1);
				}
			} else{
				if (average_duration >= new_length)
				offsets.push_back(int((average_duration-new_length+1)/2 + i*average_duration));
				else
				offsets.push_back(1);
			}
		}

		if(!ReadSegmentRGBFlowToDatum(lines_[lines_id_].first[0], lines_[lines_id_].first[1], lines_[lines_id_].second,
									 offsets, new_height, new_width, new_length, &datum)) {
			continue;
		}
		
		int offset1 = batch->data_.offset(item_id);
    	this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->TransformRgbFlow(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.rgb_flow_data_param().shuffle()){
				ShuffleVideoFlows();
			}
		}
	}
}

INSTANTIATE_CLASS(RgbFlowDataLayer);
REGISTER_LAYER_CLASS(RgbFlowData);
}
