# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:04:50 2019
Edited by Zach Ringer 2021
Edited by Ryan Thom 2024
@author: Hung Nguyen
"""

import numpy as np

def Shepherds(PaddockLength,
              SheepMatrix,
              ShepherdMatrix,
              NumberOfShepherds,
              ShepherdStep,
              SheepRadius,
              Action,
              safeDist,
              SheepGlobalCentreOfMass,
              SubGoal):
    """Calculate desired position so that the NAO can navigate effectively to
       the subgoal position.

    Args:
        PaddockLength (float): Width and Height of the environment
        SheepMatrix (np array): Sheep position matrix
        ShepherdMatrix (np array): Shepherd position matrix
        NumberOfShepherds (int): Number of shepherds
        ShepherdStep (float): Displacement per time step
        SheepRadius (float): Radius of sheep
        Action (np array): Normalised vector from Shepherd to subgoal
        safeDist (float): Radius of circle to follow without disturbing sheep
        SheepGlobalCentreOfMass (np array): Coordinates of COM
        SubGoal (np array): Subgoal position matrix

    Returns:
        tuple:
            ShepherdUpdatedMatrix (np array): Contains the desired coordinates for the shepherd
            RealAction (np array): Contains simulated desired coordinates relative to the shepherd
                                   This matrix adheres to paddock dimensions and violation of sheep
                                   radius.
    """

    # initialise the matrix to contain updated information
    ShepherdUpdatedMatrix = ShepherdMatrix

    # Vector to subgoal
    RealAction = Action

    # Loop through each shepherd
    for TheShepherd in range(0,NumberOfShepherds):

        # Define current position of NAO.
        Shepherd_CurPos = np.zeros(2)
        Shepherd_CurPos[0] = ShepherdMatrix[0,0]
        Shepherd_CurPos[1] = ShepherdMatrix[0,1]
        # Next position kept the same for now
        Shepherd_NextPos = np.zeros(2)
        Shepherd_NextPos[0] = ShepherdMatrix[0,0]
        Shepherd_NextPos[1] = ShepherdMatrix[0,1]

        # Find Distance from selected shepherd to each sheep    
        ShepherdDistanceToSheep = np.sqrt((SheepMatrix[:,0]-ShepherdMatrix[TheShepherd,0])**2 + 
                                            (SheepMatrix[:,1]-ShepherdMatrix[TheShepherd,1])**2)

        # if shepherd distance to any sheep < 3 r_a then shepherd does not move
        if np.min(ShepherdDistanceToSheep) < 3*SheepRadius:
            # Do not update position
            Shepherd_NextPos = ShepherdMatrix[TheShepherd,0:2]
            # Calculate desired position
            ShepherdUpdatedMatrix[TheShepherd,0:2] = checkNotDisturbingSheep(safeDist,Shepherd_CurPos,Shepherd_NextPos,
                                                    SheepGlobalCentreOfMass, SubGoal,ShepherdStep)
            # Final action is to not move
            RealAction = Action*0
        else:
            # Next position is towards the subgoal one timestep away
            Shepherd_NextPos = ShepherdMatrix[TheShepherd,0:2] + Action[0]*ShepherdStep
            # Calculate desired position 
            ShepherdUpdatedMatrix[TheShepherd,0:2] = checkNotDisturbingSheep(safeDist, Shepherd_CurPos, Shepherd_NextPos,
                                                    SheepGlobalCentreOfMass, SubGoal, ShepherdStep)
            RealAction = Action*0

            # Final action is the vector between desired position and current position
            RealAction[0,0] = ShepherdUpdatedMatrix[0,0] - Shepherd_CurPos[0]
            RealAction[0,1] = ShepherdUpdatedMatrix[0,1] - Shepherd_CurPos[1]
            # normalise to only keep direction
            NormOfAction = np.sqrt(RealAction[0,0]**2+RealAction[0,1]**2)
            RealAction = RealAction/NormOfAction
            # Scale for NAO's linear velocity
            RealAction = RealAction*ShepherdStep

        # Limit the movement inside the paddock:
        # TODO: Bounds should be scaled to paddock length and NAO's sway/dimensions
        #       Furthermore, the RealAction is not used to direct the shepherd, so currently
        #       NAO does not adhere to the boundaries of the environment.
        if (ShepherdUpdatedMatrix[TheShepherd,0] < -1.4):
            ShepherdUpdatedMatrix[TheShepherd,0] = -1.4
            RealAction[0,0] = 0
            RealAction[0,1] = 0

        if (ShepherdUpdatedMatrix[TheShepherd,1] < -1.4):
            ShepherdUpdatedMatrix[TheShepherd,1] = -1.4
            RealAction[0,0] = 0
            RealAction[0,1] = 0

        if (ShepherdUpdatedMatrix[TheShepherd,0] > PaddockLength+1.4):
            ShepherdUpdatedMatrix[TheShepherd,0] = PaddockLength+1.4
            RealAction[0,0] = 0
            RealAction[0,1] = 0

        if (ShepherdUpdatedMatrix[TheShepherd,1] > PaddockLength+1.4):
            ShepherdUpdatedMatrix[TheShepherd,1] = PaddockLength+1.4
            RealAction[0,0] = 0
            RealAction[0,1] = 0

    return (ShepherdUpdatedMatrix, RealAction)

def checkNotDisturbingSheep(safeDist, Shepherd_CurPos, Shepherd_NextPos, SheepGlobalCentreOfMass, SubGoal, ShepherdStep):
    """Decide on the ideal next position for the NAO. Decides best path to walk towards subgoal.

    Args:
        safeDist (float): Radius of circle to follow without disturbing sheep
        Shepherd_CurPos (np array): array containing Shepherd's current Vicon position
        Shepherd_NextPos (np array): coordinates of shepherd one time step away
        SheepGlobalCentreOfMass (np array): coordinates of COM
        SubGoal (np array): coordinates of subgoal
        ShepherdStep (float): Displacement of shepherd in a timestep based on max speed.

    Returns:
        sheedDog_UpdatePos: Array containing next point coordinates
    """
    # Shepherd Point
    SP = Shepherd_CurPos
    # COM
    LCM = SheepGlobalCentreOfMass
    # Target Point
    TP = Shepherd_NextPos
    # Max angular displacement in a timestep
    theta_step = 0.2
    # Direction between TP and COM
    TP_theta = np.arctan2(TP[1] - LCM[1], TP[0] - LCM[0])
    Pcd = SubGoal
    # Direction between subgoal and COM
    Pcd_theta = np.arctan2(Pcd[1]- LCM[1], Pcd[0] - LCM[0])

    # Distance between Shepherd and COM
    sheepDogNextDistance = np.sqrt((SP[0] - LCM[0])**2 + (SP[1] - LCM[1])**2)

    sheepDog_UpdatePos = TP

    # Check if Shepherd's distance is not violating the safe distance AND 
    # that the next point is close to the subgoal relative to the COM.
    # TODO: threshold of safeDist should be scaled to paddock length or NAO width/sway
    if ((sheepDogNextDistance < safeDist+0.6) or (sheepDogNextDistance < safeDist-0.6)) and np.abs(TP_theta - Pcd_theta) > 2 * theta_step:

        # Find shortest distance; either radius of safe circle or distance between shepherd and COM one timestep away
        r = np.minimum(safeDist, sheepDogNextDistance + ShepherdStep)
        
        # Define the closest points to SP and TP on the circle
        SP_theta = np.arctan2(SP[1] - LCM[1], SP[0] - LCM[0])
        SP_cx = r * np.cos(SP_theta) + LCM[0]
        SP_cy = r * np.sin(SP_theta) + LCM[1]

        # check if clockwise is shorter or counterclockwise

        # 1. first augment the negative values
        if (SP_theta < 0):
            SP_theta = SP_theta + 2 * np.pi

        if ((TP_theta) < 0):
            TP_theta = TP_theta + 2 * np.pi


        # 2. Decide on clockwise or counterclockwise based on distance between
        # Starting point SP and target point TP
        NP_theta = SP_theta
        # NP refers to Next point. Identify its theta first, then estimate its x, y values

        if (0 < (SP_theta - TP_theta) and (SP_theta - TP_theta) <= np.pi):
            # clockwise is shorter

            while (NP_theta >= TP_theta):
                NP_theta = NP_theta - theta_step
                NP_cx = r * np.cos(NP_theta) + LCM[0]
                NP_cy = r * np.sin(NP_theta) + LCM[1]

        elif ((SP_theta - TP_theta) > np.pi):

            # counterclockwise is shorter
            while (NP_theta <= TP_theta + 2 * np.pi):
                NP_theta = NP_theta + theta_step
                NP_cx = r * np.cos(NP_theta) + LCM[0]
                NP_cy = r * np.sin(NP_theta) + LCM[1]

        elif ((SP_theta - TP_theta) <= 0):

            # counterclockwise is shorter
            while (NP_theta <= TP_theta):
                NP_theta = NP_theta + theta_step
                NP_cx = r * np.cos(NP_theta) + LCM[0]
                NP_cy = r * np.sin(NP_theta) + LCM[1]

        sheepDog_UpdatePos = [NP_cx, NP_cy]
    return sheepDog_UpdatePos
