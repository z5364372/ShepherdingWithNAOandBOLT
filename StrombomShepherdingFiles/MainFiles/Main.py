import subprocess
import os
import sys
import dill as pickle
import base64
import numpy as np

# Insert path here
path_to_workspace = "C:/Users/ryant/OneDrive/Desktop/Thesis/NAO_Strombom_Shepherding-main"

# From here ensure this is your file structure:
"""
/MainFiles
- Main.py
/ShepherdingFiles
- Environment_Strombom.py
- findFurthestSheep.py
- NaoMain.py
- ShepherdCalculations.py
- Shepherds.py
- Sub_Env2.py
/SpheroFiles
- BoltMain.py
- BoltSheep.py
- SheepCalculations.py
- Sheeps.py
"""

# Access Environment_Strombom to initialise class variables
# TODO: Change NaoMain and BoltMain to soley use constants passed 
sys.path.append(f"{path_to_workspace}/ShepherdingFiles")


N_SHEEP = 3  # no. agents in the shepherding sim
N_EPISODE = 27  # how many trials you want to run
MAX_STEPS = 200000  # steps desired per trial

PADDOCK_LENGTH = 3.4 # Length of Vicon Arena
SHEEP_RADIUS = 0.045 # Radius of which sheep will try to avoid each other
SHEEP_SENSING_SHEPHERD_RADIUS = 0.6 # Radius of which sheep will avoid shepherd
RHO_A = 0.093 # Repulsion weight from other sheep
C = 0.0238  # Attraction weight to COM
RHO_S = 0.0227  # Repulsion weight from shepherd
H = 0.023  # Weight of Interia
SHEEP_STEP = 0.02  # delta
SHEPHERD_STEP = 0.08  # delta_s

TARGET_RADIUS = 0.22 # If COM is within this radius of the goal then end the run
SHEPHERD_RADIUS = 0.7 # Size of shepherding agent to avoid shepherd collision (if multiple are used)
TARGET_COORDS = np.array([0, 0]) # Where is the goal point
STALLING_DIST = 2.5 # How close should the shepherd approach the herd/stray sheep


N_SIZE = int(N_SHEEP-1)
N_SHEPHERDS = 1 # Number of shepherds
CASE = 1  # 1: Driving; 0: Collecting

env_vars = {"NumberOfSheep": N_SHEEP,
            "NumberOfShepherds": N_SHEPHERDS,
            "case": CASE,
            "SheepRadius": SHEEP_RADIUS,
            "SheepSensingOfShepherdRadius": SHEEP_SENSING_SHEPHERD_RADIUS,
            "PaddockLength": PADDOCK_LENGTH,
            "WeightRepellFromOtherSheep": RHO_A,
            "WeightAttractionToLocalCentreOfMassOfnNeighbours": C,
            "WeightRepellFromShepherd": RHO_S,
            "WeightOfInertia": H,
            "SheepStep": SHEEP_STEP,
            "ShepherdStep": SHEPHERD_STEP,
            "StopWhenSheepGlobalCentreOfMassDistanceToTargetIs": TARGET_RADIUS,
            "ShepherdRadius": SHEPHERD_RADIUS,
            "TargetCoordinate": TARGET_COORDS,
            "StallingFactor": STALLING_DIST,
            "N_EPISODE": N_EPISODE,
            "max_steps": MAX_STEPS,
            "NeighbourhoodSize": N_SIZE,}


serialized_env_vars = pickle.dumps(env_vars, protocol=2)
serialized_env_dict = pickle.dumps(env_vars)

# Encode the serialized object in Base64 to avoid null characters
encoded_env_vars = base64.b64encode(serialized_env_vars).decode('utf-8')
encoded_env_dict = base64.b64encode(serialized_env_dict).decode('utf-8')

# Pass the serialized object as an environment variable (in base64 to ensure it's a string)
env_vars = os.environ.copy()
env_dict = os.environ.copy()
env_vars['ENVIRONMENT'] = encoded_env_vars  # Convert bytes to a string
env_dict['ENVIRONMENT_DICT'] = encoded_env_dict  # Convert bytes to a string


# Run Python 3 script
p3_process = subprocess.Popen(
    ["C:/Users/ryant/AppData/Local/Programs/Python/Python312/python.exe",
     "C:/Users/ryant/OneDrive/Desktop/Thesis/NAO_Strombom_Shepherding-main/SpheroFiles/BoltMain.py"],
    env=env_dict
)

# Run Python 2 script
p2_process = subprocess.Popen(
    ["C:/Python27/python.exe",
     "C:/Users/ryant/OneDrive/Desktop/Thesis/NAO_Strombom_Shepherding-main/ShepherdingFiles/NaoMain.py"],
    env=env_vars
)

print("Both scripts completed.")
