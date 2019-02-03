#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import time

import cv2
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork, plot_hand_cv2
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d


def main(args):
    webcamId = 0
    try:
        if len(args) > 1 :
            webcamId = int(args[1])
    except ValueError:
        print("Invalid webcam id. Fall back to default value '" + str(webcamId) + "'.")

    # stream creation
    inputStream = cv2.VideoCapture(webcamId)
    if not inputStream.isOpened():
        print("Can not use camera with id " + str(webcamId) + ".")
        return 1

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(session)

    while True:
        _, image_raw = inputStream.read()
        image_resized = cv2.resize(image_raw, (320, 240))
        image_rgb = cv2.cvtColor(image_resized,cv2.COLOR_BGR2RGB)

        image_v = np.expand_dims((image_rgb.astype('float') / 255.0) - 0.5, 0)

        start_time = time.time()
        hand_scoremap_v, image_crop_v, scale_v, center_v,\
        keypoints_scoremap_v, keypoint_coord3d_v = session.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                             keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                            feed_dict={image_tf: image_v})
        delta_time = time.time() - start_time
        print("Inference time: " + str(delta_time))

        # post processing
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        plot_hand_cv2(image_resized, coord_hw)
        image_fullsize = cv2.resize(image_resized, (1600, 1200))

        cv2.imshow('result', image_fullsize)
        cv2.waitKey(1)

    cv2.releaseAllWindows()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
