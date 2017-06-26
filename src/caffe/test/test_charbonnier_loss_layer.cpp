

#include "caffe/layers/charbonnier_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename TypeParam>
class CharbonnierLossLayerTest : public MultiDeviceTest<TypeParam> {
private:
	typedef typename TypeParam::Dtype Dtype;
protected:
	Blob<Dtype>* top;
	Blob<Dtype>* bottom;
	vector<Blob<Dtype>*> top_vec;
	vector<Blob<Dtype>*> bottom_vec;

	CharbonnierLossLayerTest(){
		top = new Blob<Dtype>(1,1,1,1);
		bottom = new Blob<Dtype>(1,1,2,1);

		bottom->mutable_cpu_data()[0] = 3.1;
		bottom->mutable_cpu_data()[1] = 5.1;

		top_vec.push_back(top);
		bottom_vec.push_back(bottom);
	}
	virtual ~CharbonnierLossLayerTest(){
		delete top;
		delete bottom;
	}

	void TestForward() {
		//setup
		LayerParameter layer_param;
		layer_param.add_loss_weight(1);
		layer_param.mutable_charbonnier_loss_param()->set_alpha(1);
		CharbonnierLossLayer<Dtype> lossLayer(layer_param);
		lossLayer.SetUp(bottom_vec,top_vec);

		//run
		const Dtype loss1 = lossLayer.Forward(bottom_vec, top_vec);

		//setup
		LayerParameter layer_param2;
		layer_param2.add_loss_weight(1);
		layer_param2.mutable_charbonnier_loss_param()->set_alpha(0.5);
		CharbonnierLossLayer<Dtype> lossLayer2(layer_param2);
		lossLayer2.SetUp(bottom_vec,top_vec);

		//run
		const Dtype loss2 = lossLayer2.Forward(bottom_vec, top_vec);

		//evaluate
		const Dtype kErrorMargin = 1e-5;
    		EXPECT_NEAR(loss1, (3.1*3.1 + 5.1*5.1)/2, kErrorMargin);
		EXPECT_NEAR(loss2, (3.1 + 5.1)/2, kErrorMargin);
	}
};

TYPED_TEST_CASE(CharbonnierLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CharbonnierLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(CharbonnierLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  //setup
  LayerParameter layer_param;
  layer_param.add_loss_weight(1);
  layer_param.mutable_charbonnier_loss_param()->set_alpha(0.5);
  layer_param.mutable_charbonnier_loss_param()->set_beta(255);

  CharbonnierLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->bottom_vec, this->top_vec);

  //run and evaluate
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec,
      this->top_vec);

}

}  // namespace caffe
