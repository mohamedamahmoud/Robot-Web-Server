#!/usr/bin/env python3
import rospy
import time
from robo_msg.msg import Motor
from enum import IntFlag


msg = Motor()
rospy.init_node("talker", anonymous=True, disable_signals=True)
pub = rospy.Publisher("commands", Motor, queue_size=10)


class MoveType(IntFlag):
  FORWARD = 0
  BACKWARD = 1
  STOP = 2
  RIGHT = 3
  LEFT = 4

# Send ROS message
def talker(command, speed):
    if msg.command == command and msg.speed == speed:
        return
    else:
        msg.time =time.time()
        #print(msg.time)
        msg.command = command
        msg.speed = speed
        pub.publish(msg)
        #rate.sleep()
        print("Command sent: {}, speed:{}".format(command,speed))
