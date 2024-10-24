"""
Created on 2024

@author: Ryan Thom
"""

import time
from threading import Thread
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from Sheeps import Sheeps
from vicon_dssdk import ViconDataStream
import numpy as np
import math
import os
import pickle
import base64

# Convert mm to m
ViconWeight = 1000
# Max speed of BOLT
maxSpeed = 40

# BOLT serial numbers
names = ['SB-CE32', 'SB-B85A', 'SB-8427']
# names = ['SB-FD28', 'SB-7330', 'SB-5929']

def getBoltViconPositions(NumberOfSheep):
    """Get position of all the BOLTs and update sheep matrix

    Args:
        NumberOfSheep (int): Number of Sheep
    """
    # Loop through every sheep
    for i in range(NumberOfSheep):
        subject_name = "bolt" + str(i+1)
        client.GetFrame()
        global_position = client.GetSegmentGlobalTranslation(subject_name, subject_name)
        x, y, z = global_position[0]
        SheepMatrix[i][0] = (x/ViconWeight+env["PaddockLength"]/2)
        SheepMatrix[i][1] = (y/ViconWeight+env["PaddockLength"]/2)

def getNaoViconPosition():
    """Update Shepherd matrix by detecting NAO's position in Vicon
    """
    client.GetFrame()
    subject = "nao"
    global_position = client.GetSegmentGlobalTranslation(subject, subject)
    x, y, z = global_position[0]
    ShepherdMatrix[0][0] = (x/ViconWeight+env["PaddockLength"]/2)
    ShepherdMatrix[0][1] = (y/ViconWeight+env["PaddockLength"]/2)


def calcVectors():
    """Calculate repulsion vector for each BOLT
    """
    getBoltViconPositions(env["NumberOfSheep"])
    getNaoViconPosition()
    vectors = Sheeps(env["PaddockLength"], SheepMatrix,
                     env["NeighbourhoodSize"], ShepherdMatrix,
                     env["SheepRadius"], env["SheepSensingOfShepherdRadius"],
                     env["SheepStep"], env["WeightOfInertia"],
                     env["WeightRepellFromOtherSheep"],
                     env["WeightAttractionToLocalCentreOfMassOfnNeighbours"],
                     env["WeightRepellFromShepherd"])
    for bolt in range(len(SheepMatrix)):
        SheepMatrix[bolt][2:4] = vectors[bolt][2:4]

def boltMove(bolt, api):

    getBoltViconPositions(env["NumberOfSheep"])
    x0 = SheepMatrix[bolt][0]
    y0 = SheepMatrix[bolt][1]  # Initial position from matrix
    time.sleep(0.2)  # Wait to simulate movement
    getBoltViconPositions(env["NumberOfSheep"])
    x1 = SheepMatrix[bolt][0]
    y1 = SheepMatrix[bolt][1]  # New position
    calcVectors()
    # Calculate desired heading from vector
    theta_des = math.atan2(SheepMatrix[bolt][2], SheepMatrix[bolt][3])
    # Find magnitude
    mag = math.sqrt(SheepMatrix[bolt][2]**2 + SheepMatrix[bolt][3]**2)
    # Speed must be an int
    if round(maxSpeed*mag) > 0:
        # Convert heading into degrees
        theta_des = round(math.degrees(theta_des))
        # Change direction of BOLT
        api.set_heading(theta_des)
        # Set linear velocity
        api.set_speed(round(maxSpeed*mag))

    ### Uncomment below for sensor fusion to adjust for drift
    # elif math.sqrt((x1 - x0)**2 + (y1-y0)**2) >= 0.1:
        # theta_abs = math.degrees(math.atan2(x1-x0, y1-y0))
        # print(f"Vicon Angle: {theta_abs}")
        # theta_current = api.get_heading()
        # print(f"Bolt Angle: {theta_current}")
        # theta_delta = theta_current-theta_abs
        # print(f"Delta Angle: {theta_delta}")
        # theta_des = theta_des-theta_delta
        # print(f"Adjusted Angle: {theta_des}")
    else:
        api.set_speed(0)

class SpheroController(Thread):
    """Sphero Class to control each robot with the same script

    Args:
        Thread (thread obj): Each instance of threading
    """
    def __init__(self, toy, bolt_id):
        """Initialising function

        Args:
            toy (sphero_edu obj): Used to command BOLTs
            bolt_id (int): index of BOLT or sheep
        """
        super().__init__()
        self.toy = toy
        self.bolt_id = bolt_id

    def run(self):
        """Main function to run the BOLTs
        """
        with SpheroEduAPI(self.toy) as api:
            print(f"Starting {self.toy.name}")
            while True:
                boltMove(self.bolt_id, api)


def main():
    """BOLT Main script
    """
    try:
        # names = ['SB-FD28', 'SB-7330', 'SB-2743']  # Names of the robots
        # Scan for BOLTs. NOTE: Increasing timeout can help find them
        toys = scanner.find_toys(timeout=11, toy_names=names)
        print(toys)
        threads = []
        # Start each BOLT in a different thread using the same class
        for i, toy in enumerate(toys):
            controller = SpheroController(toy, i)
            threads.append(controller)
            controller.start()

        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    # Initalise Vicon
    client = ViconDataStream.Client()
    vicon_server_ip = '192.168.68.54'
    vicon_server_port = '801'
    client.Connect(f"{vicon_server_ip}:{vicon_server_port}")
    client.SetBufferSize(3)
    client.EnableSegmentData()
    print("Vicon Initialised Bolts")

    # Get the Base64-encoded serialized object from the environment variable
    encoded_object = os.environ['ENVIRONMENT_DICT']

    # Decode it back into bytes
    serialized_object = base64.b64decode(encoded_object)

    # Deserialise it
    env = pickle.loads(serialized_object)

    SheepMatrix = np.zeros([env["NumberOfSheep"], 5])
    ShepherdMatrix = np.zeros([env["NumberOfShepherds"], 5])
    initRobotPositionMatrix = getNaoViconPosition()
    SheepPositions = getBoltViconPositions(env["NumberOfSheep"])

    main()
