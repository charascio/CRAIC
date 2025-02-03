#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

if __name__ == '__main__':
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('RedImage', Image, queue_size=10)
    bridge = CvBridge()
    cap = cv2.VideoCapture(0)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_pub.publish(ros_image)
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
