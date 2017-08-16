#!/usr/bin/env python

import os, sys
import caffe
import numpy as np
import cv2
import utils

def OpticalFlowPrediction(image_pairs, network):
    # # Display the network structure
    # for ele in network.blobs.items():
    #   print ele
    network.blobs['data'].data[...] = np.transpose(image_pairs, [0,3,1,2])
    output = network.forward()
    estimated_flow = network.blobs['predict_flow2'].data
    # print estimated_flow.shape

    return np.transpose(estimated_flow, [0,2,3,1])


def main():

    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    model_def_file = './deploy.prototxt'
    model_file = './MotionNet.caffemodel'
    flownet = caffe.Net(model_def_file, model_file, caffe.TEST)

    data_dir = "~/UCF101/frames"
    dest_dir = "~/UCF101/unsupFlow"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    class_list = os.listdir(data_dir)
    class_list.sort()

    # get the mean file
    batch_size = 10
    batch_id = 0
    img_list = []
    flow_x_path_list = []
    flow_y_path_list = []

    height = 256
    width = 320
    output_width = 340
    lower_bound = -20
    upper_bound = 20
    multiplier = 20
    intensity_range = upper_bound - lower_bound
    R = np.ones((height, width)) * 104.0
    G = np.ones((height, width)) * 117.0
    B = np.ones((height, width)) * 123.0
    mean_file = np.stack((R,G,B),axis=2)

    for class_name in class_list:
        class_dir = os.path.join(data_dir, class_name)
        scene_list = os.listdir(class_dir)
        scene_list.sort()
        dest_class_dir = os.path.join(dest_dir, class_name)

        if not os.path.exists(dest_class_dir):
            os.makedirs(dest_class_dir)
        for scene in scene_list:
            scene_dir = os.path.join(class_dir, scene)
            frame_list = os.listdir(scene_dir)
            frame_list.sort()
            dest_scene_dir = os.path.join(dest_class_dir, scene)
            if not os.path.exists(dest_scene_dir):
                os.makedirs(dest_scene_dir)

            for frame_id in xrange(len(frame_list) - 1):
                if batch_id == 0:
                    start_path = os.path.join(scene_dir, frame_list[frame_id])
                    start_img = cv2.imread(start_path, cv2.IMREAD_UNCHANGED)
                    start_img = cv2.resize(start_img, (width, height))
                    start_img = (start_img - mean_file)/255.0
                    img_list.append(np.expand_dims(start_img, 0))

                next_path = os.path.join(scene_dir, frame_list[frame_id+1])
                next_img = cv2.imread(next_path, cv2.IMREAD_UNCHANGED)
                next_img = cv2.resize(next_img, (width, height))
                next_img = (next_img - mean_file)/255.0
                img_list.append(np.expand_dims(next_img, 0))

                flow_x_name = 'flow_x_{0:04d}.jpg'.format(frame_id+1)
                flow_x_path = os.path.join(dest_scene_dir, flow_x_name)
                flow_x_path_list.append(flow_x_path)

                flow_y_name = 'flow_y_{0:04d}.jpg'.format(frame_id+1)
                flow_y_path = os.path.join(dest_scene_dir, flow_y_name)
                flow_y_path_list.append(flow_y_path)

                batch_id += 1

                if batch_id == batch_size:
                    flownet_input = np.concatenate(img_list, axis=3)
                    # print(flownet_input.shape)
                    flow = OpticalFlowPrediction(flownet_input, flownet).squeeze() * multiplier
                    # print(flow.shape)
                    
                    for flow_num in xrange(batch_size):
                        flow_x = flow[:,:,0 + flow_num*2]
                        flow_y = flow[:,:,1 + flow_num*2]

                        np.clip(flow_x, lower_bound, upper_bound)
                        flow_x = (flow_x - lower_bound) * 255 / intensity_range
                        flow_x = cv2.resize(flow_x, (output_width, height))
                        cv2.imwrite(flow_x_path_list[flow_num], flow_x.astype(int))

                        np.clip(flow_y, lower_bound, upper_bound)
                        flow_y = (flow_y - lower_bound) * 255 / intensity_range
                        flow_y = cv2.resize(flow_y, (output_width, height))
                        cv2.imwrite(flow_y_path_list[flow_num], flow_y.astype(int))

                    # clear the buffer
                    batch_id = 0
                    flow_x_path_list = []
                    flow_y_path_list = []
                    img_list = []

                    continue

                if frame_id == len(frame_list) - 2 and batch_id < batch_size:
                    # the last several frames, not enough for a batch of 10, so pad data 
                    cur_len = len(img_list)
                    pad_len = batch_size + 1 - len(img_list)
                    for pad_index in range(pad_len):
                        img_list.append(img_list[-1])

                    flownet_input = np.concatenate(img_list, axis=3)
                    # print(flownet_input.shape)
                    flow = OpticalFlowPrediction(flownet_input, flownet).squeeze() * multiplier
                    # print(flow.shape)
                    
                    for flow_num in xrange(cur_len-1):
                        flow_x = flow[:,:,0 + flow_num*2]
                        flow_y = flow[:,:,1 + flow_num*2]
                        # print flow_x.shape, flow_y.shape

                        np.clip(flow_x, lower_bound, upper_bound)
                        flow_x = (flow_x - lower_bound) * 255 / intensity_range
                        flow_x = cv2.resize(flow_x, (output_width, height))
                        cv2.imwrite(flow_x_path_list[flow_num], flow_x.astype(int))

                        np.clip(flow_y, lower_bound, upper_bound)
                        flow_y = (flow_y - lower_bound) * 255 / intensity_range
                        flow_y = cv2.resize(flow_y, (output_width, height))
                        cv2.imwrite(flow_y_path_list[flow_num], flow_y.astype(int))

                    # this folder done
                    batch_id = 0
                    flow_x_path_list = []
                    flow_y_path_list = []
                    img_list = []

            print("scene %s is done" % (scene))
        print("class %s is done" % class_name)


if __name__ == "__main__":
    main()
