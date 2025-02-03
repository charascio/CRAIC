#! /usr/bin/env python3

import sys
sys.path.append("/home/wql/catkin_ws/src/task1_red_recognition/scripts")

import rospy
from red_recognition import detect_red_objects, draw_rect
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    red_objects_info = detect_red_objects(frame)
    draw_rect(frame, red_objects_info)
    # cv2.imshow("Red Object Detection", frame)
    # cv2.waitKey(1)
    ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    pub.publish(ros_image)

if __name__ == '__main__':
    rospy.init_node("red_pub")
    sub = rospy.Subscriber("RedImage", Image, callback, queue_size=10)
    bridge = CvBridge()
    pub = rospy.Publisher("RedImageProcessed", Image, queue_size=10)
    rospy.spin()
