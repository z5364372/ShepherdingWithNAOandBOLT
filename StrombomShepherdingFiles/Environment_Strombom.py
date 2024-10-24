"""
Created on Thu Feb 21 13:28:32 2019
Edited by Zach Ringer 2021
Edited by Ryan Thom 2024
@author: Hung Nguyen
"""
import numpy as np
from Shepherds import Shepherds
import Sub_Env2
from findFurthestSheep import findFurthestSheep
import matplotlib.pyplot as plt


class Environment:
    """
    This class is the major 'hub' for all of the Strombom rules based shepherding functions. This class has been,
    simplified to work in a UGV shepherding demonstration. If you wish to edit it to have more parameters for
    simulation use please refer to Zach Ringer's version https://github.com/RingaDinga01/NAO_Strombom_Shepherding.
    """
    def __init__(self,
                 NumberOfSheep,
                 NumberOfShepherds,
                 case,
                 SheepRadius,
                 SheepSensingOfShepherdRadius,
                 PaddockLength,
                 WeightRepellFromOtherSheep,
                 WeightAttractionToLocalCentreOfMassOfnNeighbours,
                 WeightRepellFromShepherd,
                 WeightOfInertia,
                 SheepStep,
                 ShepherdStep,
                 StopWhenSheepGlobalCentreOfMassDistanceToTargetIs,
                 ShepherdRadius, 
                 TargetCoordinate,
                 StallingFactor,
                 SheepMatrix=None,
                 ShepherdMatrix=None
                 ):
        """Initialiser function. All editable constants are defined in the Main.py script which is pushed to NAO
        and the BOLTs.

        Args:
            NumberOfSheep (int): Number of sheep in the arena
            NumberOfShepherds (int): Number of shepherds
            case (int): 1 = Driving, 0 = collecting
            SheepRadius (float): Radius to avoid sheep collision
            SheepSensingOfShepherdRadius (float): Repulsion range from shepherd
            PaddockLength (float): Dimensions of arena
            WeightRepellFromOtherSheep (float): rho_A
            WeightAttractionToLocalCentreOfMassOfnNeighbours (float): c
            WeightRepellFromShepherd (float): rho_S
            WeightOfInertia (float): h
            SheepStep (float): Speed of sheep REMOVE
            ShepherdStep (float): displacement of shepherd
            StopWhenSheepGlobalCentreOfMassDistanceToTargetIs (float): Target radius
            ShepherdRadius (float): Radius of shepherd to avoid collisions with each other
            SheepMatrix (5x5 array): Contains sheep information. Defaults to None.
            ShepherdMatrix (5x5 array): Contains shepherd information.
            TargetCoordinate (1x2 array): Contains coordinates of goal.
            StallingFactor (float): How close the shepherd gets within the sheep
        """
        # Global General Parameters
        self.NumberOfSheep = NumberOfSheep
        self.NumberOfShepherds = NumberOfShepherds
        self.case = case

        # Strombom Parameters
        self.PaddockLength = PaddockLength
        self.SheepRadius = SheepRadius
        self.SheepSensingOfShepherdRadius = SheepSensingOfShepherdRadius
        self.WeightRepellFromOtherSheep = WeightRepellFromOtherSheep
        self.WeightAttractionToLocalCentreOfMassOfnNeighbours = WeightAttractionToLocalCentreOfMassOfnNeighbours
        self.WeightRepellFromShepherd = WeightRepellFromShepherd
        self.WeightOfInertia = WeightOfInertia

        # Implementation Parameters
        self.ShepherdStep = ShepherdStep
        self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs = StopWhenSheepGlobalCentreOfMassDistanceToTargetIs
        self.ShepherdRadius = ShepherdRadius
        self.TargetCoordinate = TargetCoordinate
        self.StallingFactor = StallingFactor

        # Sheep Variables
        self.CMAL_ALL = None
        self.SheepGlobalCentreOfMass = None # COM Coordinates
        self.IndexOfFurthestSheep = None # Index to identify the furthest sheep
        self.MaximumSheepDistanceToGlobalCentreOfMass = self.WeightRepellFromOtherSheep * (self.NumberOfSheep**(2/float(3))) # f(n)

        # Shepherd Variables
        self.ShepherdGlobalCentreOfMass = None # COM of shepherd (unused if 1 in play)
        self.R1 = self.MaximumSheepDistanceToGlobalCentreOfMass
        self.R2 = self.SheepSensingOfShepherdRadius # R1 and R2 used for NAO controller
        self.SubGoal = None # Subgoal coordinates (driving or collecting pos)
        self.sheepDog_t1_2LCM_MinRadius = 0 # Radius of circle to move along when navigating to subgoal

    def reset(self, initRobotPositionMatrix, SheepPositions):
        """Method used to reset the enviornment and initialise the shepherd and sheep

        Args:
            initRobotPositionMatrix (np array): Contains Vicon coordinates of NAO at the start of the run
            SheepPositions (np array): Contains Vicon coordinates of all BOLTs at the start of the run

        Returns:
            tuple: 
                InitialState (np array): Important vectors depending on if the shepherd is collecting or driving
                AreFurthestSheepCollected (int): Boolean to determine if sheep are collected or not. 
        """
        # Initialise all position matricies
        self.SheepMatrix, self.ShepherdMatrix = Sub_Env2.create_Env(self.NumberOfShepherds, self.NumberOfSheep,
                                                                    initRobotPositionMatrix, SheepPositions)
        
        self.CMAL_ALL = np.zeros([self.NumberOfSheep,8])

        # Calculating Initial COMs
        self.SheepGlobalCentreOfMass = np.array([np.mean(self.SheepMatrix[:,0]),np.mean(self.SheepMatrix[:,1])])  # GCM of sheep objects
        self.ShepherdGlobalCentreOfMass = np.array([np.mean(self.ShepherdMatrix[:,0]),np.mean(self.ShepherdMatrix[:,1])])  # GCM of shepherd objects


        self.IndexOfFurthestSheep, AreFurthestSheepCollected = findFurthestSheep(self.SheepMatrix,
                                                                                self.SheepGlobalCentreOfMass,
                                                                                self.NumberOfShepherds,
                                                                                self.MaximumSheepDistanceToGlobalCentreOfMass)     
        InitialState = self.cal_State()

        return (InitialState, AreFurthestSheepCollected)


    def stepRealAction(self, action, CurrentNaoPos):
        """Additional step function used to get NAOs actual desired position from the Strombom rules

        Args:
            action (np array): Normalised vector from Shepherd to subgoal
            CurrentNaoPos (np array): coordinates of NAO's position

        Returns:
            tuple:
                RealAction (np array): Contains simulated desired coordinates relative to the shepherd
                                       This matrix adheres to paddock dimensions and violation of sheep
                                       radius.
                ShepherdMatrix (np array): Contains the desired coordinates for the shepherd
        """
        # Distance from the CM to furthest sheep
        furthestDist = np.sqrt((self.SheepGlobalCentreOfMass[0]-self.SheepMatrix[self.IndexOfFurthestSheep, 0])**2 +
                               (self.SheepGlobalCentreOfMass[1]-self.SheepMatrix[self.IndexOfFurthestSheep, 1])**2)

        # outter (3rd) circle radius
        self.sheepDog_t1_2LCM_MinRadius = np.maximum(furthestDist, self.R1) + self.R2
        # Update shepherd matrix with Vicon data
        self.ShepherdMatrix[0, 0] = CurrentNaoPos[0, 0]
        self.ShepherdMatrix[0, 1] = CurrentNaoPos[1, 0]

        self.ShepherdMatrix, RealAction = Shepherds(self.PaddockLength, self.SheepMatrix,
                                                    self.ShepherdMatrix, self.NumberOfShepherds,
                                                    self.ShepherdStep, self.SheepRadius,
                                                    action,
                                                    self.sheepDog_t1_2LCM_MinRadius, self.SheepGlobalCentreOfMass, self.SubGoal)

        return (RealAction, self.ShepherdMatrix)


    def step(self, action, NextNaoPos):
        """Step function used to update Strombom calculations and matricies. Check's if run has completed or not

        Args:
            action (np array): Normalised Vector of Shepherd to Subgoal
            NextNaoPos (np array): Vicon position of NAO

        Returns:
            tuple:
                NextState (np array): Contains various vectors depending if driving on collecting
                Terminate (int): Check if shepherding has completed or not
                ShepherdMatrix (np array): Shepherd Matrix with updated Vicon positions
                SheepMatrix (np array): Updated Sheep Matrix with Vicon positions
                FurthestSheep (np array): Position matrix of furthest sheep
                AreFurthestSheepCollected (int): Check if sheep are within f(n)
                infor (np array): Information matrix containing various details about the shepherding demonstration.
                                  Used mostly for plotting and saving the environment for analysis.
        """
        # Distance from the CM to furthest sheep
        self.SheepMatrix = np.asarray(self.SheepMatrix)
        self.ShepherdMatrix = np.asarray(self.ShepherdMatrix)
        furthestDist = np.sqrt((self.SheepGlobalCentreOfMass[0]-self.SheepMatrix[self.IndexOfFurthestSheep][0])**2 +
                               (self.SheepGlobalCentreOfMass[1]-self.SheepMatrix[self.IndexOfFurthestSheep][1])**2)

        # outter (3rd) circle radius
        self.sheepDog_t1_2LCM_MinRadius = np.maximum(furthestDist, self.R1) + self.R2
        # Update ShepherdMatrix
        self.ShepherdMatrix[0,0] = NextNaoPos[0,0]
        self.ShepherdMatrix[0,1] = NextNaoPos[1,0]

        # Recompute Sheep Centre of mass
        self.SheepGlobalCentreOfMass = np.array([np.mean(self.SheepMatrix[:, [0]]),
                                                 np.mean(self.SheepMatrix[:, [1]])])  # GCM of sheep objects
        self.ShepherdGlobalCentreOfMass = np.array([np.mean(self.ShepherdMatrix[:, [0]]),
                                                    np.mean(self.ShepherdMatrix[:, [1]])])  # GCM of shepherd objects

        # Find furthest sheep and check if they are collected
        self.IndexOfFurthestSheep, AreFurthestSheepCollected = findFurthestSheep(self.SheepMatrix,
                                                                                 self.SheepGlobalCentreOfMass,
                                                                                 self.NumberOfShepherds,
                                                                                 self.MaximumSheepDistanceToGlobalCentreOfMass)
        # Checking terminate
        Terminate, SubGoalPosition = self.check_terminate()

        # Calculating Next State
        NextState = self.cal_State()
        # Information matrix for plotting
        infor = (self.SheepMatrix[0, [0]], self.SheepMatrix[0, [1]],
                 self.SheepMatrix[1, [0]], self.SheepMatrix[1, [1]],
                 self.SheepMatrix[2, [0]], self.SheepMatrix[2, [1]],
                 self.SheepGlobalCentreOfMass[0], self.SheepGlobalCentreOfMass[1],
                 self.SheepMatrix[self.IndexOfFurthestSheep, 0],
                 self.SheepMatrix[self.IndexOfFurthestSheep, 1],
                 SubGoalPosition[0], SubGoalPosition[1])

        return (NextState, Terminate, self.ShepherdMatrix, self.SheepMatrix,
                self.SheepMatrix[self.IndexOfFurthestSheep, :],
                AreFurthestSheepCollected, infor)


    def Strombom_action(self,driving):
        """Depending on whether NAO is determined to be driving or collection,
        the relevant subgoal calculation methods will be run

        Args:
            driving (int): 1: Driving; 0: collecting

        Returns:
            tuple:
                action (np array): Normalised Vector of Shepherd to Subgoal
                SubGoal (np array): Global coordinates of subgoal
        """
        if driving == 1:
            # Calculate driving position (rho_d)
            self.SubGoal = self.cal_subgoal_behindcenter()
        else:
            # Calculate collecting position (rho_c)
            self.SubGoal = self.cal_subgoal_behindfurthest()
        action = self.SubGoal - self.ShepherdMatrix[:,0:2]        
        NormOfAction = np.sqrt(action[:,0]**2+action[:,1]**2)
        action = action/NormOfAction
        return (action, self.SubGoal)

    def cal_State(self):
        """Calculates important vectors
        Position of Dog (PD), center of sheeps (CS), furthest sheep(FS), target (T)
        Or vector of PD to CS, PD to FS, CS to T

        Returns:
            State (np Array): Contains various vectors depending if driving on collecting
        """
        
        State = np.zeros(4)

        if self.case == 1: # driving
            State[0] = self.SheepGlobalCentreOfMass[0] - self.ShepherdMatrix[0,0] #PD to CS (x)
            State[1] = self.SheepGlobalCentreOfMass[1] - self.ShepherdMatrix[0,1] #PD to CS (y)
            State[2] = self.TargetCoordinate[0] - self.SheepGlobalCentreOfMass[0] #CS to T (x)
            State[3] = self.TargetCoordinate[0] - self.SheepGlobalCentreOfMass[1] #CS to T (y)
        else: # collecting
            State[0] = self.SheepGlobalCentreOfMass[0] - self.ShepherdMatrix[0,0] #PD to CS (x)
            State[1] = self.SheepGlobalCentreOfMass[1] - self.ShepherdMatrix[0,1] #PD to CS (y)          
            State[2] = self.SheepMatrix[self.IndexOfFurthestSheep,0] - self.SheepGlobalCentreOfMass[0] #CS to FS (x)
            State[3] = self.SheepMatrix[self.IndexOfFurthestSheep,1] - self.SheepGlobalCentreOfMass[1] #CS to FS (y)

        State = State/self.PaddockLength

        return State
        

    def check_terminate(self):
        """Function used to terminate the algorithm when certain conditions are met such 
            as the GCM being moved to within a pre-defined distance from the target location


        Returns:
            tuple:
                terminate (int): 1: shepherding completed
                SubGoalPosition (np array): position of subgoal
        """
        terminate = 0  # Not achieving the target
        if (self.case == 1): # If driving
            SubGoalPosition = self.cal_subgoal_behindcenter()
        else:
            SubGoalPosition = self.cal_subgoal_behindfurthest()
            
        dist_PD_SubGoal = np.sqrt((self.ShepherdMatrix[0,0]-SubGoalPosition[0])**2+
                                    (self.ShepherdMatrix[0,1]-SubGoalPosition[1])**2) # Distance GCM sheep to target
        if (dist_PD_SubGoal <= self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs):
            terminate = 1 #Achieving the target
        
        return (terminate, SubGoalPosition)


    def cal_subgoal_behindcenter(self):
        """Calculates the subgoal direction vector and position when the 
           Shepherd is deemed to be driving the herd because all sheep are aggregated

        Returns:
            PositionBehindCenter (np array): Contains coordinates of driving position
        """
        # Vector TG -> CM
        DirectionFromTargetToGlobalCentreOfMass = np.array([self.SheepGlobalCentreOfMass[0]-self.TargetCoordinate[0],
                                                            self.SheepGlobalCentreOfMass[1]-self.TargetCoordinate[1]])
        # Distance TG -> CM
        NormOfDirectionFromTargetToGlobalCentreOfMass = np.sqrt(DirectionFromTargetToGlobalCentreOfMass[0]**2 + DirectionFromTargetToGlobalCentreOfMass[1]**2)
        # Norm TG -> CM (Preserve direction only)
        NormalisedDirectionFromTargetToGlobalCentreOfMass = DirectionFromTargetToGlobalCentreOfMass / NormOfDirectionFromTargetToGlobalCentreOfMass
        # Distance from COM using Stalling Distance and Strombom's equation.
        # Found in the direction away from target relevant to COM
        PositionBehindCenterFromTarget = NormalisedDirectionFromTargetToGlobalCentreOfMass * (NormOfDirectionFromTargetToGlobalCentreOfMass + (self.SheepRadius*(np.sqrt(self.NumberOfSheep)*self.StallingFactor)))#* np.sqrt(self.NumberOfSheep)+1.5)#-0.1 works well
        # Final position normalised to global coordinates
        PositionBehindCenterFromTarget[0] += self.TargetCoordinate[0]
        PositionBehindCenterFromTarget[1] += self.TargetCoordinate[1]
        return PositionBehindCenterFromTarget
    

    def cal_subgoal_behindfurthest(self):
        """Calculates the subgoal direction vector and position when the Shepherd is deemed to 
           be collcting because one or more sheep have dispersed from the herd


        Returns:
            PositionBehindFurthestSheep (np array): Coordinates of collecting position
        """
        # COM coordinates
        COM = np.array([self.SheepGlobalCentreOfMass[0], self.SheepGlobalCentreOfMass[1]])
        CS_FS_x = self.SheepMatrix[self.IndexOfFurthestSheep,0] - self.SheepGlobalCentreOfMass[0] # CS to FS x
        CS_FS_y = self.SheepMatrix[self.IndexOfFurthestSheep,1] - self.SheepGlobalCentreOfMass[1] # CS to FS y
        # COM -> FS
        DirectionFromGlobalCentreOfMassToFurthestSheep = np.array([CS_FS_x, CS_FS_y])
        # Distance of COM -> FS
        NormOfDirectionFromGlobalCentreOfMassToFurthestSheep = np.sqrt(DirectionFromGlobalCentreOfMassToFurthestSheep[0]**2 + DirectionFromGlobalCentreOfMassToFurthestSheep[1]**2)
        # Normalised directional component of COM -> FS
        NormalisedDirectionFromGlobalCentreOfMassToFurthestSheep = DirectionFromGlobalCentreOfMassToFurthestSheep / NormOfDirectionFromGlobalCentreOfMassToFurthestSheep
        # Distance from COM using Stalling Distance and Strombom's equation.
        # Relative to COM
        DirectionFromGlobalCentreOfMassToPositionBehindFurthestSheep = NormalisedDirectionFromGlobalCentreOfMassToFurthestSheep*(NormOfDirectionFromGlobalCentreOfMassToFurthestSheep + (self.SheepRadius*self.StallingFactor))#-0.25 works good
        # Normalise for global coordinates
        PositionBehindFurthestSheep = COM + DirectionFromGlobalCentreOfMassToPositionBehindFurthestSheep

        return PositionBehindFurthestSheep


    def view(self,subgoal,driving,ShepherdCurrPos):
        """This function is used to plot the simulation in real-time

        Args:
            subgoal (np array): global position of subgoal
            driving (int): Is NAO driving or collecting
            ShepherdCurrPos (np array): Global position of NAO
        """
        # Plotting---------------------------------------------------------
        self.SheepGlobalCentreOfMass = np.array([np.mean(self.SheepMatrix[:,0]),np.mean(self.SheepMatrix[:,1])])  # GCM of sheep objects
        self.ShepherdGlobalCentreOfMass = np.array([np.mean(self.ShepherdMatrix[:,0]),np.mean(self.ShepherdMatrix[:,1])]) # GCM of shepherd objects

        fHandler = plt.figure(1)

        fHandler.Color = 'white'
        fHandler.MenuBar = 'none'
        fHandler.ToolBar = 'none'      
        fHandler.NumberTitle = 'off'
        ax = fHandler.gca()
        ax.cla() # clear things for fresh plot

        # change default range 
        ax.set_xlim((-self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs-0.5, self.PaddockLength+self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs+0.5))
        ax.set_ylim((-self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs-0.5, self.PaddockLength+self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs+0.5))

        ax.plot(self.SheepMatrix[:,0],self.SheepMatrix[:,1],'k.',markersize=10) # plot sheeps 
        ax.plot(self.ShepherdMatrix[:,0],self.ShepherdMatrix[:,1],'b*',markersize=10)# plot shepherd
        ax.plot(self.SheepGlobalCentreOfMass[0],self.SheepGlobalCentreOfMass[1],'ro',markersize=10); # plot GCM of sheep

        #Plotting the subgoal for testing
        subgoalx = subgoal[0]
        subgoaly = subgoal[1]
        ax.plot(subgoalx,subgoaly,'bs',markersize=8)
        #plt.plot(ShepherdGlobalCentreOfMass(1,1),ShepherdGlobalCentreOfMass(1,2),'rd','markersize',10); % plot GCM of shepherds
        ax.plot(self.TargetCoordinate[0],self.TargetCoordinate[1],'gp',markersize=10) # plot target point
        
        circle=plt.Circle([self.TargetCoordinate[0],self.TargetCoordinate[1]],self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs,color='r')
        ax.add_artist(circle)
        if driving==1:
            circle2 = plt.Circle((self.SheepGlobalCentreOfMass[0],self.SheepGlobalCentreOfMass[1]), (self.SheepSensingOfShepherdRadius), color='b', fill=False)
            ax.add_artist(circle2)
        else:
            innerDrivingCircle = plt.Circle((self.SheepMatrix[self.IndexOfFurthestSheep,0], self.SheepMatrix[self.IndexOfFurthestSheep,1]), (self.SheepSensingOfShepherdRadius), color='m', fill=False)
            ax.add_artist(innerDrivingCircle)
        circle3 = plt.Circle((self.SheepGlobalCentreOfMass[0],self.SheepGlobalCentreOfMass[1]), (self.sheepDog_t1_2LCM_MinRadius), color='r', fill=False)
        ax.add_artist(circle3)
        drivingCircle = plt.Circle((self.SheepGlobalCentreOfMass[0],self.SheepGlobalCentreOfMass[1]), self.MaximumSheepDistanceToGlobalCentreOfMass, color='g', fill=False)
        ax.add_artist(drivingCircle)
        
        ax.plot([0,0],[0,self.PaddockLength],'b-')
        
        ax.plot([0,self.PaddockLength],[0,0],'b-')
        ax.plot([self.PaddockLength,0],[self.PaddockLength,self.PaddockLength],'b-')
        ax.plot([self.PaddockLength,self.PaddockLength],[self.PaddockLength,0],'b-')


        #Plot Next Shepherd Step
        ShepherdCurrPosx =ShepherdCurrPos[0,0]
        ShepherdCurrPosy =ShepherdCurrPos[1,0]
        ax.plot(ShepherdCurrPosx,ShepherdCurrPosy,'r*',markersize=10)
        plt.xlabel('Paddock Length')
        plt.ylabel('Paddock Height')
        plt.pause(0.05)
        plt.show(block=False)
