#ifndef PHOTOMETRIC_AUGMENTATION_LAYER_HPP_
#define PHOTOMETRIC_AUGMENTATION_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
notes

gaussian noise, sigma uniformly sampled from 0-0.4
contrast: [-0.8, 0.4]
multiplicative color changes: [0.05,2]
gamma: [0.7 1.5]
additive brightness: gaussian sigma 0.2

*/

template <typename Dtype>
class PhotometricAugmentationLayer: public Layer<Dtype> {

public:
	explicit PhotometricAugmentationLayer(const LayerParameter& param): Layer<Dtype>(param) {      }
	~PhotometricAugmentationLayer();

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "PhotometricAugmentation"; }
	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline bool EqualNumBottomTopBlobs() const { return true; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	bool hasMeans;
	Blob<Dtype> means;

	vector<Blob<Dtype>*> noise;
	Blob<Dtype> contrast;
	Blob<Dtype> colour;
	Blob<Dtype> gamma;
	Blob<Dtype> brightness;
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
