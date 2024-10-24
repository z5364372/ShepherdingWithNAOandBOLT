# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:54:22 2019
Edited by Zach Ringer 2021
Edited by Ryan Thom 2024
@author: z5095790, Hung Nguyen

This script finds the furtherst sheep from the sheep COM and also the distance to the shepherd
"""

import numpy as np

def findFurthestSheep(SheepMatrix, SheepGlobalCentreOfMass,
                      NumberOfShepherds, MaximumSheepDistanceToGlobalCentreOfMass):
    """Find furthest sheep from COM so that it can be collected. Also determine if all sheep are collected.

    Args:
        SheepMatrix (np Array): Sheep position matrix
        SheepGlobalCentreOfMass (np Array): COM position Matrix
        NumberOfShepherds (int): number of shepherds
        MaximumSheepDistanceToGlobalCentreOfMass (float): f(n)

    Returns:
        tuple:
            IndexOfFurthestSheep (int): index of furthest sheep
            AreFurthestSheepCollected (int): Boolean flag to determine if sheep are collected or not.
    """

    NumberOfSheep = len(SheepMatrix)

    # Array containing all distances to COM for each sheep
    SheepDistanceToGlobalCentreOfMass = np.sqrt((SheepMatrix[:,0]-SheepGlobalCentreOfMass[0])**2+(SheepMatrix[:,1]-SheepGlobalCentreOfMass[1])**2)

    # Find and sort the distance between LCM of the sheep and every sheep
    GCMDistanceToSheepWithIndex = np.zeros([NumberOfSheep,2])
    GCMDistanceToSheepWithIndex[:,0] = SheepDistanceToGlobalCentreOfMass
    GCMDistanceToSheepWithIndex[:,1] = SheepMatrix[:,4]
    # Sort the to find largest distance
    SortedGCMDistanceToSheepWithIndex = GCMDistanceToSheepWithIndex[GCMDistanceToSheepWithIndex[:,0].argsort()[::-1]]

    # Find the index of the furthest sheep to LCM
    if NumberOfShepherds > 1:
        IndexOfFurthestSheep = SortedGCMDistanceToSheepWithIndex[0:NumberOfShepherds,1].astype(int)
    else:
        IndexOfFurthestSheep = SortedGCMDistanceToSheepWithIndex[0,1].astype(int)

    # find the collection status of the furthest sheep
    AreFurthestSheepCollected = 0

    if SortedGCMDistanceToSheepWithIndex[0, 0] <= MaximumSheepDistanceToGlobalCentreOfMass:
        AreFurthestSheepCollected = 1

    return (IndexOfFurthestSheep, AreFurthestSheepCollected)
