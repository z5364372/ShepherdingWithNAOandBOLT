"""
Created on Fri Dec 20 12:05:22 2019
Edited by Zach Ringer 2021
Edited by Ryan Thom 2024
@author: Hung Nguyen

This script is responsible for initialising the strombom environment
"""


import numpy as np


def create_Env(NumberOfShepherds, NumberOfSheep, 
               initRobotPositionMatrix, SheepPositions):
    """Initalise Sheep and Shepherd matricies

    Args:
        NumberOfShepherds (int): Number of shepherds
        NumberOfSheep (int): Number of sheep in the arena
        initRobotPositionMatrix (np array): Contains Vicon coordinates of NAO at the start of the run
        SheepPositions (np array): Contains Vicon coordinates of all BOLTs at the start of the run

    Returns:
        list:
            SheepMatrix (np array): Sheep position matrix
    """

    SheepMatrix = np.zeros([NumberOfSheep, 5])  # initial population of sheep **A matrix of size OBJECTSx5

    # create the shepherd matrix
    ShepherdMatrix = np.zeros([NumberOfShepherds, 5])  # initial population of shepherds **A matrix of size OBJECTSx5
    # Denote position of first shepherd to location of NAO
    ShepherdMatrix[0, 0] = initRobotPositionMatrix[0, 0]
    ShepherdMatrix[0, 1] = initRobotPositionMatrix[0, 1]

    # Denote position of Sheep to position of BOLTs
    for i in range(NumberOfSheep):
        SheepMatrix[i, [0]] = SheepPositions[i][0]
        SheepMatrix[i, [1]] = SheepPositions[i][1]

    # Initialise Sheep Initial Directions Angle [-pi,pi]
    SheepMatrix[:, 2] = np.pi - np.random.rand(len(SheepMatrix[:, 2]))*2*np.pi  # 1 - because just having one column

    # Add the index of each sheep into the matrix
    SheepMatrix[:, 4] = np.arange(0, len(SheepMatrix[:, 4]), 1)

    # Initialise Shepherds Initial Directions Angle [-pi,pi]
    ShepherdMatrix[:, 2] = np.pi - np.random.rand(NumberOfShepherds)*2*np.pi

    # Add the index of each shepherd into the matrix
    ShepherdMatrix[:, 4] = np.arange(0, len(ShepherdMatrix[:, 4]), 1)

    return list([SheepMatrix, ShepherdMatrix])
