import numpy as np

d = 0.166  # Calculated from positions

quad_params = {
    # Inertial properties
    'mass': 0.826,       # kg
    'Ixx':  0.0047,     # kg*m^2
    'Iyy':  0.005,     # kg*m^2
    'Izz':  0.0074,     # kg*m^2
    'Ixy':  0.0,        # kg*m^2
    'Iyz':  0.0,        # kg*m^2 
    'Ixz':  0.0,        # kg*m^2
    'arm_length': d,    # meters
    'com': np.array([0.0, 0.0, 0.0]),  # Center of mass position in body frame

    #           x
    #           ^
    #      mot0+| mot1-
    #           |
    #     y<----+-----
    #           |
    #      mot3-| mot2+
    #    
    
    # Geometric properties, all vectors are relative to the center of mass.
    'num_rotors': 4,                        # for looping over each actuator
    'rotor_radius': 0.10,                   # rotor radius, in meters for visualization
    'rotor_pos': {  
                    'r1': d*np.array([ np.sin(np.pi/4),   np.cos(np.pi/4), 0]),    # Location of Rotor 1 (front left), meters
                    'r2': d*np.array([ np.sin(np.pi/4),  -np.cos(np.pi/4), 0]),    # Location of Rotor 2 (front right), meters
                    'r3': d*np.array([-np.sin(np.pi/4),  -np.cos(np.pi/4), 0]),    # Location of Rotor 3 (back right), meters
                    'r4': d*np.array([-np.sin(np.pi/4),   np.cos(np.pi/4), 0]),    # Location of Rotor 4 (back left), meters
                 },

    'rotor_directions': np.array([1,-1,1,-1]),  # This dictates the direction of the torque for each motor. 
    'rotor_efficiency': np.array([1.0,1.0,1.0,1.0]),  # This dictates the efficiency of each motor. 

    'rI': np.array([0,0,0]), # location of the IMU sensor, meters

    # drag coefficient for each axis
    'cd1x': 0.62,
    'cd1y': 0.62,
    'cd1z': 0.62,

    'cdz_h': 0.00,


    # Frame aerodynamic properties - set to 0 as per yaml
    'c_Dx': 0.00,  # parasitic drag in body x axis, N/(m/s)**2
    'c_Dy': 0.00,  # parasitic drag in body y axis, N/(m/s)**2
    'c_Dz': 0.00,  # parasitic drag in body z axis, N/(m/s)**2

    # Rotor properties
    'k_eta': 7.64e-6,      # thrust coefficient N/(rad/s)**2 from thrust_map[0]
    'k_m': 7.64e-6 * 0.0140,  # yaw moment coefficient Nm/(rad/s)**2 (k_eta * kappa)
    'k_d': 1.19e-04,          # rotor drag coefficient N/(rad*m/s**2) = kg/rad
    'k_z': 2.32e-04,          # induced inflow coefficient N/(rad*m/s**2) = kg/rad
    'k_flap': 0.0,            # Flapping moment coefficient Nm/(rad*m/s**2) = kg*m/rad

    # Motor properties
    'tau_m': 0.005,             # motor response time, seconds
    'rotor_speed_min': 0.0,   # rad/s
    'rotor_speed_max': 1000.0,  # rad/s
    'motor_noise_std': 50,      # rad/s
}
