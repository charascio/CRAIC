#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ocs2_msgs.msg import mode_schedule
import sys
import tty
import termios

# enum ModeNumber {
#   FLY = 0,
#   LCONTACT = 1,
#   RCONTACT = 2,
#   STANCE = 3,
# };

# 用于读取键盘输入的函数
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

if __name__ == '__main__':
    rospy.init_node("simple_publisher")

    pub1 = rospy.Publisher("/humanoid_mpc_mode_schedule", mode_schedule, queue_size=10)
    pub2 = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    # 初始化 mode_schedule 消息
    flag = mode_schedule()
    flag.eventTimes = [0.0, 0.45, 0.9]  # 使用列表初始化
    flag.modeSequence = [1, 2]           # 使用列表初始化

    # 发布 flag 消息五次
    for i in range(5):
        pub1.publish(flag)
        rospy.loginfo("Published mode schedule: %s", flag)
        rospy.sleep(0.1)  # 等待一段时间，避免发布过快

    rate = rospy.Rate(10)

    velocity = Twist()

    while not rospy.is_shutdown():
        # 读取键盘输入
        key = getch()
        
        # 打印键盘输入
        rospy.loginfo("Key pressed: %s", key)

        # 根据键盘输入设置速度
        if key == 'w':
            velocity.linear.x = 0.1
        elif key == 's':
            velocity.linear.x = -0.1
        elif key == 'x':  # 停止
            velocity.linear.x = 0.0
        else:
            continue  # 如果不是w、s或x，则继续循环

        # 发布速度消息
        pub2.publish(velocity)
        rate.sleep()
