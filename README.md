Hidden Two-Stream Convolutional Networks for Action Recognition
============================

This is the Caffe implementation of the "Hidden Two-Stream Convolutional Networks for Action Recognition". You can refer to paper for more details at [Arxiv](https://arxiv.org/abs/1704.00389).


Dependencies
=========

OpenCV 3 (Installation can be refered [here](https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04))

Tested on Ubuntu 16.04 with Titan X GPU, CUDNN 5.1

Compiling
=========

To get started, first compile caffe, by configuring a

    "Makefile.config" 

then make with 

    $ make -j 6 all

Training
========

(this assumes you compiled the code sucessfully) 

Here, we take UCF101 split 1 as an example. 

First, go to folder, 

    cd models/ucf101_split1_unsup_end
    
Then change the `FRAME_PATH` in `train_rgb_split1.txt` and `val_rgb_split1.txt` to where you store the extracted video frames,  

    /FRAME_PATH/WallPushups/v_WallPushups_g21_c06 111 98

This follows the format as in [TSN](https://github.com/yjxiong/temporal-segment-networks). `111` indicates the number of frames of that video clip, and `98` represents the action label. For more details about how to construct file list for training and validation, we refer you to [here](https://github.com/yjxiong/temporal-segment-networks#construct-file-lists-for-training-and-validation).

Then you need to download the initialization models (pre-trained temporal stream CNN stacked upon pre-trained MotionNet), 

[UCF101 split1](https://drive.google.com/open?id=0B-bJpXHBmFWDNnZ2TnE3cVZTNVU) 

Then tune the parameters in `end_train_val.prototxt` and `end_solver.prototxt` as you need, or leave as it is. 

Finally, you can simply run

    ../../build/tools/caffe train -solver=end_solver.prototxt -weights=ucf101_split1_vgg16_init.caffemodel


NOTE: It is highly likely that you may get better performance than us if you carefully tune the hyper-params such as loss weights, learning rate etc. 

Testing
========

(this assumes you compiled the code sucessfully) 

First, download our trained models:

[UCF101 split 1](https://drive.google.com/open?id=0B-bJpXHBmFWDamFiUmp0UGpwY2c) [UCF101 split 2](https://drive.google.com/open?id=0B-bJpXHBmFWDVlpULU5tcmdGaGs) [UCF101 split 3](https://drive.google.com/open?id=0B-bJpXHBmFWDNmozVDlPSTFWdEE) 

[HMDB51 split 1](https://drive.google.com/open?id=0B-bJpXHBmFWDUER6OUdyVmNyenM) [HMDB51 split 2](https://drive.google.com/open?id=0B-bJpXHBmFWDcmxVZmxyUWVJbzQ) [HMDB51 split 3](https://drive.google.com/open?id=0B-bJpXHBmFWDenZpWlFqNm0yMnM) 

Then go to this folder

    cd models/ucf101_split1_unsup_end/eval_ucf101

Then run

    python demo_hidden.py

But maybe you need to set paths correctly in `demo_hidden.py` before you run it, like `model_def_file` and `model_file`. And also change the `FRAME_PATH` in `testlist01_with_labels.txt`. 

After you get both spatial and hidden predictions, the late fusion code is in folder `./test`, run `late_fusion.m` to get the final two stream predictions.


MotionNet
=========

The training and testing code of MotionNet is in folder

	cd models/multiframe_MotionNet

The pretraied model can be downloaded at [MotionNet](https://drive.google.com/open?id=0B-bJpXHBmFWDVU5DRTY4Ym02TFE).


Misc
====================

1. There is a chance that you may get a little bit higher or lower accuracy on UCF101 and HMDB51 than the numbers reported in our paper, even using our provided trained models. This is normal because your extracted video frames may not be the same as ours, and the quality of image has an impact on the final performance. Thus, no need to raise an issue unless the performance gap is large, e.g. larger than 1%. 

2. Since there are so many losses to compute, you may encounter model divergence in the very beginning of the training. You can simply reduce learning rate first to get a good initialization, and then back on track. Or you just rerun training several times. 


TODO
====================

- [ ] Experiment on large-scale action datasets, like Sports-1M and Kinetics 


License and Citation
====================

Please cite this paper in your publications if you use this code or precomputed results for your research:

    @article{hidden_ar_zhu_2017,
      title={{Hidden Two-Stream Convolutional Networks for Action Recognition}},
      author={Yi Zhu and Zhenzhong Lan and Shawn Newsam and Alexander G. Hauptmann},
      journal={arXiv preprint arXiv:1704.00389},
      year={2017}
    }

Related Projects
====================

[GuidedNet](https://github.com/bryanyzhu/GuidedNet): Guided Optical Flow Learning

[Two_Stream Pytorch](https://github.com/bryanyzhu/two-stream-pytorch): PyTorch implementation of two-stream networks for video action recognition


Acknowledgement
====================

The code base is borrowed from [TSN](https://github.com/yjxiong/temporal-segment-networks), [DispNet](https://lmb.informatik.uni-freiburg.de/resources/software.php) and [UnsupFlownet](http://scs.ryerson.ca/~jjyu/). Thanks for open sourcing the code.