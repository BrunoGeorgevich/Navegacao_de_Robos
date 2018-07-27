import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import numpy as np


def callback(data):
    diffs = []
    for index in range(1, len(data.ranges)):
        last_cell = data.ranges[index - 1]
        curr_cell = data.ranges[index]
        diffs.append(curr_cell - last_cell)

    if np.var(data.ranges) < 1.5:
        vels = {"linear": {"x": 0, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": 2}}
        move(vels)
    else:
        vels = {"linear": {"x": 20, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": 0}}
        move(vels)


def move(vels):
    vel_msg = Twist()
    vel_msg.linear.x = vels["linear"]['x']
    vel_msg.linear.y = vels["linear"]['y']
    vel_msg.linear.z = vels["linear"]['z']
    vel_msg.angular.x = vels["angular"]['x']
    vel_msg.angular.y = vels["angular"]['y']
    vel_msg.angular.z = vels["angular"]['z']

    velocity_publisher.publish(vel_msg)


stage = {"width": 40, "height": 40}
robot = {"x": 0, "y": 0, "z": 0, "theta": 0}

rospy.init_node('robot', anonymous=True)
velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rospy.Subscriber("base_scan", LaserScan, callback)
rospy.spin()
