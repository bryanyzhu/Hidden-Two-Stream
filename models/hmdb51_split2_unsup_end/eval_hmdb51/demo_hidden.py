#!/usr/bin/env python

import os, sys
import numpy as np
import caffe
import math
import cv2
import scipy.io as sio
import h5py

from HiddenTemporalPrediction import HiddenTemporalPrediction

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():

    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # spatial prediction
    model_def_file = '../stack_motionnet_vgg16_deploy.prototxt'
    model_file = '../logs_end/hmdb51_split2_vgg16_hidden.caffemodel'
    FRAME_PATH = "TODO"
    spatial_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    val_file = "./testlist02.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print "we got %d test videos" % len(val_list)

    start_frame = 0
    num_categories = 51
    feature_layer = 'fc8_vgg16'
    spatial_mean_file = './rgb_mean.mat'
    dims = (len(val_list), num_categories)
    predict_results_before = np.zeros(shape=dims, dtype=np.float64)
    predict_results = np.zeros(shape=dims, dtype=np.float64)

    correct = 0
    line_id = 0
    spatial_results_before = {}
    spatial_results = {}

    for line in val_list:
        line_info = line.split(" ")
        input_video_dir_part = line_info[0]
        input_video_dir = os.path.join(FRAME_PATH, input_video_dir_part[:-4])
        input_video_label = int(line_info[1])

        spatial_prediction = HiddenTemporalPrediction(
                input_video_dir,
                spatial_mean_file,
                spatial_net,
                num_categories,
                feature_layer,
                start_frame)
        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        avg_spatial_pred = np.asarray(softmax(avg_spatial_pred_fc8))
        predict_label = np.argmax(avg_spatial_pred)

        predict_results_before[line_id, :] = avg_spatial_pred_fc8
        predict_results[line_id, :] = avg_spatial_pred

        print input_video_dir
        print input_video_label-1, predict_label

        line_id += 1
        if predict_label == input_video_label-1:
            correct += 1

    print correct
    print "prediction accuracy is %4.4f" % (float(correct)/len(val_list))

    spatial_results_before["hidden_prediction_before"] = predict_results_before
    spatial_results["hidden_prediction"] = predict_results

    sio.savemat("./hmdb51_split2_hidden_before.mat", spatial_results_before)
    sio.savemat("./hmdb51_split2_hidden.mat", spatial_results)
    

if __name__ == "__main__":
    main()
