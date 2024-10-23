import time
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color
from qLearning import LightTime

# File name to write results to
log = open("sphero_log_braitenberg_method_x.csv", 'w')
# csv headings
log.write("time_t, dist_t, time_l, light\n")

threshold = 1000 # Light level required to satisfy finding the source
c = 200 # Constant to determine angle weight
speed = 70 # Operating speed of Sphero BOLT

def collision_detection(api):
    '''
    Detect if robot is stuck on wall, and adjust it to start moving again

    :param: api: SphereoEDUApi Object (the robot being controlled)
    
    :return: None
    '''
    # Check if the robot is tilted up or sideways too much
    if api.get_orientation()['pitch'] >= 65 or abs(api.get_orientation()['roll']) >= 15:
        api.stop_roll() # Stop movement
        # Spin until the robot is leveled out
        while (api.get_orientation()['pitch'] >= 30):
            api.spin(45, 0.1)


def adjust_vel(api):
    '''
    Adjust the velocity, or angle at which the robot is rolling to achieve a circular
    path.

    :param: api: SphereoEDUApi Object (the robot being controlled)
    
    :return: None
    '''
    # Get light level
    lum = api.get_luminosity()['ambient_light']
    weight = c/lum # The bigger the luminosity the smaller the weight
    api.set_speed(speed) # Scalar speed is constant
    api.spin(360//weight, 1) # The smaller the weight the bigger the turn is

def data_log(api):
    '''
    Gather data and put into csv

    :param: api: SphereoEDUApi Object (the robot being controlled)
    
    :return: None
    '''
    dist = api.get_distance() # distance coord (cm)
    light = abs(api.get_luminosity()['ambient_light']) # Light level (lum)
    if light >= 800: # Time under light
        LightTime = time.time_ns()
    else:
        LightTime = 0
    # Write data to file
    data = str(time.time_ns()) + ", " + str(dist) + ", " + str(LightTime) + ", " + str(light) + "\n"
    log.write(data)

def main():
    '''
    Main operating function

    :param: None
    
    :return: None
    '''
    toys = scanner.find_toys(toy_names=['SB-FD28']) # Find robot
    with SpheroEduAPI(toys[0]) as droid: # Start programming robot

        # Set back LED to white
        droid.set_main_led(Color(255, 255, 255))
        # Adjust velocity of robot
        adjust_vel(droid)
        LightTime = 0
 
        try:
            TotalTimeStart = time.time()
            while(TotalTimeStart - time.time() > -90): # End program after 90 seconds
                print(TotalTimeStart-time.time())
                adjust_vel(droid)
                collision_detection(droid) # Constantly check if stuck
                light = abs(droid.get_luminosity()['ambient_light']) # Read light level
                data_log(droid)

            droid.stop_roll() # Stop program
        except KeyboardInterrupt:
            print('Interrupted')
            
# Initialise program
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')