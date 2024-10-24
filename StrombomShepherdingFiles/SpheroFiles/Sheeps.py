# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:19:43 2019
Edited by Zach Ringer 2021
Edited by Ryan Thom 2024
@author: Hung Nguyen
"""

from SheepCalculations import SheepCalculations
import numpy as np

def Sheeps(PaddockLength, SheepMatrix, NeighbourhoodSize,
           ShepherdMatrix, SheepRadius, SheepSensingOfShepherdRadius,
           SheepStep, WeightOfInertia, WeightRepellFromOtherSheep,
           WeightAttractionToLocalCentreOfMassOfnNeighbours,
           WeightRepellFromShepherd):
    """Calculate the Force vector which the sheep will follow.

    Args:
        PaddockLength (float): Dimensions of arena
        SheepMatrix (np array): Position and previous direction of each sheep
        NeighbourhoodSize (int): Number of neighbors sheep
        ShepherdMatrix (np array): Shepherd position matrix
        SheepRadius (float): Radius of sheep
        SheepSensingOfShepherdRadius (float): Sensor range
        SheepStep (float): Sheep speed
        WeightOfInertia (float): Momentum weight
        WeightRepellFromOtherSheep (float): Repulsion from other sheep
        WeightAttractionToLocalCentreOfMassOfnNeighbours (float): attraction to COM
        WeightRepellFromShepherd (float): repulsion form shepherd
    
    Return:
        UpdatedSheepMatrix (np array): Updated matrix containing force vector
    """

    NumberOfSheep = len(SheepMatrix)

    UpdatedSheepMatrix = np.zeros([NumberOfSheep, 5]) # For simultaneous update
    CMAL_ALL = None
    for i in range(0, NumberOfSheep): # Go through every sheep object.
        # Calculate force vectors on sheep
        CMAL = SheepCalculations(i,SheepMatrix,NeighbourhoodSize,ShepherdMatrix,SheepRadius,SheepSensingOfShepherdRadius) 
        # Assign vectors
        RepulsionDirectionFromOtherSheep    = CMAL[0,:]	# Direction of repulsion from other sheep

        RepulsionDirectionFromShepherds     = CMAL[1,:]	# Direction of repulsion from shepherds

        AttractionDirectionToOtherSheep     = CMAL[2,:]	# Direction of attraction to other sheep.

        DistanceToShepherds                 = CMAL[3,0]	# Distance to shepherds.

        # NumberOfNearestNeighbours           = CMAL(3,1)	# Number of nearest
        # neighbours - Not Used in This Function
        # Assign vectors into class variable
        if CMAL_ALL is None:
            CMAL_ALL = CMAL.reshape([1,8])
        else:
            CMAL_ALL = np.vstack([CMAL_ALL,CMAL.reshape([1,8])])

        # Determine direction from previous timestep        
        PreviousDirection = SheepMatrix[i,2:4]  # Direction in the previous time step

        if (DistanceToShepherds < SheepSensingOfShepherdRadius): # If reacting to the shepherd
        
            # Calculate new direction using forces and weights
            NewDirection=WeightOfInertia*PreviousDirection+WeightRepellFromOtherSheep*RepulsionDirectionFromOtherSheep+WeightRepellFromShepherd*RepulsionDirectionFromShepherds+WeightAttractionToLocalCentreOfMassOfnNeighbours*AttractionDirectionToOtherSheep # New direction of sheep i

            # Normalise direction
            if (NewDirection[0] == 0 and NewDirection[1] == 0):
                NormalisedNewDirection = np.array([0,0])
            else:
                NormalisedNewDirection=(NewDirection/np.sqrt(NewDirection[0]**2+NewDirection[1]**2)) #Normalized direction of sheep i

            # New directional angle
            # TODO: Experiment using non-normalised direction to include magnitude of vector
            UpdatedSheepMatrix[i,2:4] = NormalisedNewDirection[0:2]

            # Preserve the index of the object
            UpdatedSheepMatrix[i,4]=SheepMatrix[i,4]

        else:
            SheepMatrix[i,2:4] = 0
            UpdatedSheepMatrix[i,2:4]=SheepMatrix[i,2:4]

    return UpdatedSheepMatrix
