#! /usr/bin/env python
# -*- encoding: UTF-8 -*-
"""
Created on 2021
Edited by Ryan Thom 2024
@author: Zach Ringer
"""

import sys
import qi
import argparse

import almath
import math
import time

import numpy as np
import datetime
import pandas as pd
from Environment_Strombom import Environment
from vicon_dssdk import ViconDataStream

import os
import dill as pickle
import base64


def main(session):
    """Main function to control NAO and Strombom functions

    Args:
        session (NAOqi obj): Allows direct control of NAO via APIs
    """
    # Define env as a global variable so that all functions can access its constants and functions
    global env

    # env = Environment(env_vars["NumberOfSheep"], env_vars["NeighbourhoodSize"], env_vars["NumberOfShepherds"], env_vars["case"])
    env = Environment(env_vars["NumberOfSheep"],
                      env_vars["NumberOfShepherds"],
                      env_vars["case"],
                      env_vars["SheepRadius"],
                      env_vars["SheepSensingOfShepherdRadius"],
                      env_vars["PaddockLength"],
                      env_vars["WeightRepellFromOtherSheep"],
                      env_vars["WeightAttractionToLocalCentreOfMassOfnNeighbours"],
                      env_vars["WeightRepellFromShepherd"],
                      env_vars["WeightOfInertia"],
                      env_vars["SheepStep"],
                      env_vars["ShepherdStep"],
                      env_vars["StopWhenSheepGlobalCentreOfMassDistanceToTargetIs"],
                      env_vars["ShepherdRadius"],
                      env_vars["TargetCoordinate"],
                      env_vars["StallingFactor"])
    
    # Get the API services ALMotion & ALRobotPosture.
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    # Getting initial robot position
    initRobotPosition = getViconPosition()
    NaoGlobal = np.matrix([initRobotPosition.x, initRobotPosition.y])

    motion_service.stopMove()  # make NAO stop before beginning a new run

    # Initalize the arms for better walk stability
    leftArmEnable = True
    rightArmEnable = True
    motion_service.setMoveArmsEnabled(leftArmEnable, rightArmEnable)

    # Send robot to Stand Init
    posture_service.goToPosture("StandInit", 0.5)

    # Initialize the move
    motion_service.moveInit()

    # Clock setup to track the total time to run a simulation trial
    t_start = time.time()  # Start the clock
    t = 0  # set current time to 0 before loop begins

    # Create some text files to write data (Uncomment to save data)
    # NaoPos = open('NaoPosition.txt', 'w')
    # ShepherdPos = open('ShepherdPos.txt', 'w')
    # TimeFile = open('Time.txt', 'w')
    # SpeedFile = open('Speed.txt', 'w')

    # Create the header for the CSV file to save data to
    header = ["Ep", "Step", "ShepherdX", "ShepherdY", "ShepherdXvel",
              "ShepherdYvel", "Sheep1x", "Sheep1y", "Sheep2x", "Sheep2y",
              "Sheep3x", "Sheep3y", "SheepGCMx", "SheepGCMy", "Furthestx",
              "FurthestY", "SubGoalX", "SubGoalY", "time"]
    filename = "Collecting05_1.csv"
    with open(filename, 'w') as f:
        pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False,
                                            header=True)
    t = time.time()-t_start

    # This code is required for controlling the sheep and shepherd in Hungs
    # Strombom algorithm
    
    # Loop through each episode
    for episode_i in range(env_vars["N_EPISODE"]):
        # Convert mm to m
        ViconWeight = 1000
        # Detect sheep positions
        SheepPositions = np.zeros((env.NumberOfSheep, 2))
        for i in range(env.NumberOfSheep):
            subject_name = "bolt" + str(i+1)
            client.GetFrame()
            global_position = client.GetSegmentGlobalTranslation(subject_name, subject_name)
            x, y, z = global_position[0]
            SheepPositions[i][0] = (x/ViconWeight+env.PaddockLength/2)
            SheepPositions[i][1] = (y/ViconWeight+env.PaddockLength/2)
        s_t, case = env.reset(NaoGlobal, SheepPositions)

        # Loop through number of timesteps
        for step in range(env_vars["max_steps"]):
            reset = 0
            # Find position of NAO
            RobotPosition = getViconPosition()
            CurrentNaoPos = np.matrix([[RobotPosition.x], [RobotPosition.y]])
            # Find collecting or driving position
            a_t, subgoal = env.Strombom_action(case)
            # Getting Next Shepherd Position and Velocity
            real_action, ShepherdMatrix = env.stepRealAction(a_t, CurrentNaoPos)

            # Define Desired Global Position (x right y up)
            posx = ShepherdMatrix[0, 0]
            posy = ShepherdMatrix[0, 1]

            NaoGlobal = NaoMove(posx, posy, motion_service, RobotPosition,
                                real_action, reset)
            # Allow time for NAO to move
            time.sleep(0.05)
            # Get position of NAO again
            RobotPosition = getViconPosition()
            CurrentNaoPos1 = np.matrix([[RobotPosition.x], [RobotPosition.y]])
            # Update sheep matrix
            getBoltViconPositions(env_vars["NumberOfSheep"])
            # Refresh Strombom shepherding matricies and variables
            s_t1, done, ShepherdMatrix, SheepMatrix, SheepFurthest, case, infor = env.step(a_t, CurrentNaoPos1)

            # Desired Coordinates
            posx = ShepherdMatrix[0, 0]
            posy = ShepherdMatrix[0, 1]

            t = time.time()-t_start

            # Save the data to a csv file
            save_data = np.hstack([episode_i+1, step+1, posx, posy,
                                   real_action[0, 0], real_action[0, 1],
                                   infor[0], infor[1], infor[2], infor[3],
                                   infor[4], infor[5], infor[6], infor[7],
                                   infor[8], infor[9], infor[10], infor[11], t]).reshape(1, 19)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False,
                                               header=False)

            # View the simulation figure in real-time
            env.view(subgoal, case, CurrentNaoPos)

            # If the COM is within 0.2x0.2m from target,
            # end the sim episode/trial
            # TODO: Scale this with parameters defined in Main.py
            if (infor[6] < 0.1 and infor[7] < 0.1):
                break


def NaoMove(posx, posy, motion_service, RobotPosition, real_action, setpoint):
    """NAO movement controller

    Args:
        posx (float): Desired x position
        posy (float): Desired y position
        motion_service (API): Movement object to control NAO
        RobotPosition (np array): Matrix containing NAO coordinates
        real_action (np array): Matrix containing desired NAO vector to desired position
        setpoint (int): 0: Shepherding in progress; 1: End shepherding run

    Returns:
        tuple:
            NaoGlobal (np array): Coordinates of NAO
            dFromNextPoint (np array): Distance to next point 
    """
    # Walk Parameters
    Kp = 0.30  # Position Gain
    Kthy = 0.50  # Turn Rate Gain

    NaoGlobal = np.matrix([[RobotPosition.x], [RobotPosition.y]])
    NaoGlobalx = NaoGlobal[0, 0]
    NaoGlobaly = NaoGlobal[1, 0]

    # Setting Global Velocity
    if setpoint == 0:
        xGdesDeriv = real_action[0, 0]
        yGdesDeriv = real_action[0, 1]
    else:
        xGdesDeriv = 0
        yGdesDeriv = 0

    # Get NAOs current heading (Use VICON GLOBAL Heading with Vicon)
    cTheta = RobotPosition.theta  # Grab NAOs Heading

    # Create matrix with Global difference in position (x right y up)
    PosErrorMat2 = np.matrix([[posx-NaoGlobalx], [posy-NaoGlobaly]])

    # Create the desired velocity vector in Global (x right y up)
    VdesGlobal = np.matrix([[xGdesDeriv], [yGdesDeriv]])

    # Create Vc command Velocity in NAO Global (x up y left)
    Vcc = VdesGlobal+Kp*(PosErrorMat2)
    Vccx = Vcc[0, 0]
    Vccy = Vcc[1, 0]

    # Compute the Desired Global Thy
    ThyDes = math.atan2(Vccy, Vccx)

    # Compute Desired turn angle error
    thyAng3 = ThyDes-cTheta

    # Deal with the heading flip around the unit circle
    if thyAng3 > np.pi:
        finalHeading = thyAng3-(2*np.pi)
    elif thyAng3 < -np.pi:
        finalHeading = thyAng3+(2*np.pi)
    else:
        finalHeading = thyAng3

    # Compute Desired turn rate
    turnRate = Kthy*finalHeading

    # Compute Desired Forward Speed
    vcxsquare = np.square(Vccx)
    vcysquare = np.square(Vccy)
    Sdes = math.sqrt(vcxsquare+vcysquare)

    # If sDes (Forward Speed) is greater than 0.08m/s then set it to 0.08m/s as
    # this is approximately NAOs fastest forward movement speed
    # (10.5cms/s actually)
    if Sdes > 0.08:
        Sdes = 0.08

    # Check if turn rate is greater than 0.53rad/s in both directions
    # (NAO Max Turn Rate)
    if turnRate > 0.5:
        turnRate = 0.5
    if turnRate < -0.5:
        turnRate = -0.5

    # SpeedFile.write("%s\n" % np.column_stack((Sdes,t)))
    dFromNextPoint = np.square(PosErrorMat2[0, 0]+PosErrorMat2[1, 0])

    # Limit speed of robot if it is close
    if dFromNextPoint > 0.05:
        motion_service.move(Sdes, 0, turnRate)
    else:
        Sdes = Sdes*0.5
        motion_service.move(Sdes, 0, turnRate)

    # Set Robot Movement Speed and Turn Rate
    motion_service.move(Sdes, 0, turnRate)

    return(NaoGlobal, dFromNextPoint)


def changeHeading(angle):
    """Adjust the heading for NAO's perspective, as NAO is 90 degrees out of alignment

    Args:
        angle (float): Heading from Vicon

    Returns:
        angle (float): Heading adjusted for NAO's POV
    """
    if (angle <= 0 and angle > -math.pi/2):
        angle = angle + math.pi/2
    elif (angle <= -math.pi/2 and angle >= -math.pi):
        angle = angle + math.pi/2
    elif (angle <= math.pi and angle >= math.pi/2):
        angle = angle - (3*math.pi)/2
    elif (angle < math.pi/2 and angle > 0):
        angle = angle + math.pi/2
    return angle

def getBoltViconPositions(NumberOfSheep):
    """Update sheep matrix to contain BOLT positions from Vicon

    Args:
        NumberOfSheep (int): Number of sheep
    """
    ViconWeight = 1000
    for i in range(NumberOfSheep):
        subject_name = "bolt" + str(i+1)
        client.GetFrame()
        global_position = client.GetSegmentGlobalTranslation(subject_name, subject_name)
        x, y, z = global_position[0]
        env.SheepMatrix[i][0] = (x/ViconWeight+env.PaddockLength/2)
        env.SheepMatrix[i][1] = (y/ViconWeight+env.PaddockLength/2)


def getViconPosition():
    """Get position and heading of NAO from Vicon

    Returns:
        almath Pose2D array: Localisation of NAO robot with x, y and heading
    """
    # Convert mm to m
    ViconWeight = 1000

    # Read Vicon frame
    client.GetFrame()
    global_position = client.GetSegmentGlobalTranslation(subject_name,
                                                         segment_name)
    global_orientation = client.GetSegmentGlobalRotationEulerXYZ(subject_name,
                                                                 segment_name)
    x, y, z = global_position[0]
    roll, pitch, yaw = global_orientation[0]

    return almath.Pose2D((x/ViconWeight+env.PaddockLength/2),
                         (y/ViconWeight+env.PaddockLength/2), changeHeading(yaw))


def __init__(self, position):
    self.position = position

# Initialising Arguments
if __name__ == "__main__":
    # Parse the NAO IP address
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.68.52",
                        help="Robot IP address. On robot or Local Naoqi:use '127.0.0.1'")  # 127.0.0.1 or 192.168.1.22 "169.254.237.71"
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    # Start NAOqi session
    args = parser.parse_args()
    session = qi.Session()
    # Connect to NAO
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port "
               + str(args.port) + ".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    # Connect to Vicon
    client = ViconDataStream.Client()
    client.Connect("192.168.68.54:801")
    client.SetBufferSize(3)
    client.EnableSegmentData()
    has_frame = False

    while not has_frame:
        client.GetFrame()
        has_frame = True

    print("Vicon Initiated for Nao")

    # Vicon object name for NAO
    subject_name = "nao"
    segment_name = "nao"

    # Get the Base64-encoded serialised object from the environment variable (env object)
    encoded_object = os.environ['ENVIRONMENT']

    # Decode it back into bytes
    serialized_object = base64.b64decode(encoded_object)

    # Deserialise it
    env_vars = pickle.loads(serialized_object)

    # Start main function
    main(session)
