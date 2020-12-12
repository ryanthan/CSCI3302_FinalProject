"""
CSCI 3302: Introduction to Robtics Fall 2020
Jorge Ortiz Venegas, Ryan Than, Sage Garrett, Sarah Schwallier, William Culkin
Maze Search + Rescue 
Robot Controller.
"""

import copy
from controller import Supervisor
import numpy as np
import math

# Based from lab 5
robot_node = None
supervisor = None
target_node = None

# Based from labs 3-5
def init_supervisor():
    global supervisor, robot_node, target_node

    # create the Supervisor instance.
    supervisor = Supervisor()

    # do this once only
    root = supervisor.getRoot() 
    root_children_field = root.getField("children") 
    robot_node = None
    target_node = None
    for idx in range(root_children_field.getCount()):
        if root_children_field.getMFNode(idx).getDef() == "EPUCK":
            robot_node = root_children_field.getMFNode(idx)
        if root_children_field.getMFNode(idx).getDef() == "Goal":
            target_node = root_children_field.getMFNode(idx) 

    start_translation = copy.copy(robot_node.getField("translation").getSFVec3f())
    start_rotation = copy.copy(robot_node.getField("rotation").getSFRotation())

# Based from lab 5
def supervisor_get_target_pose():
    target_position = np.array(target_node.getField("translation").getSFVec3f())
    target_pose = np.array([target_position[0], 1. - target_position[2], target_node.getField("rotation").getSFRotation()[3] + math.pi/2.])
    # print("Target pose relative to robot: %s" % (str(target_pose)))
    return target_pose

# Based from lab 5
def supervisor_get_robot_pose():
    robot_position = np.array(robot_node.getField("translation").getSFVec3f())
    robot_pose = np.array([robot_position[0], 1. - robot_position[2], robot_node.getField("rotation").getSFRotation()[3]+math.pi/2])
    return robot_pose