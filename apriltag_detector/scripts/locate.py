#!/home/project/anaconda3/envs/pytorch/bin/python
#coding=utf-8


import pyrealsense2 as rs
import cv2
import apriltag
from loguru import logger
import numpy as np
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge
from referee_msgs.msg import Apriltag_info

fx = 386.494
fy = 385.552
ppx = 324.572
ppy = 243.29

if __name__ == "__main__":

    rospy.init_node("locate")

    image_pub = rospy.Publisher('/color', Image, queue_size=1)
    loc_pub = rospy.Publisher('/location', Apriltag_info, queue_size=1)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）
    align_to = rs.stream.color

    alignedFs = rs.align(align_to)

    profile = pipeline.start(cfg)

    tag = Apriltag_info()

    while True:
        start_time = cv2.getTickCount()

        fs = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(fs)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame or not color_frame:
            continue
        origin_img = np.asanyarray(color_frame.get_data())
        deep_img = np.asanyarray(depth_frame.get_data())

        bridge = CvBridge()
        image_message = bridge.cv2_to_imgmsg(origin_img, encoding="bgr8")

        image_pub.publish(image_message)
        gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        
        if len(results) != 0:
            for r in results:
                tag.tag_num = len(results)
                tag.tag_pos_z = float(deep_img[int(r.center[1])][int(r.center[0])]) * 0.001  # 框的深度值
                tag.tag_pos_x = float((r.center[0] - ppx) / fx * tag.tag_pos_z)
                tag.tag_pos_y = float((r.center[1] - ppx) / fx * tag.tag_pos_z)
                loc_pub.publish(tag)

        end_time = cv2.getTickCount()
        time_took = (end_time - start_time) / cv2.getTickFrequency()

        logger.info("FPS: {}".format(1 / time_took))
                