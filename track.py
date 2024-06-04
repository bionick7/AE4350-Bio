import numpy as np

class Track:
    def __init__(self) -> None:
        pass

    def get_input(self, states: np.ndarray) -> np.ndarray:
        ''' Returns sensor readings evaluated at a certain point 
            positions: Nx4 array of generation states
            returns: Nx? array of sensor positions corresponding to positions'''
        N = len(states)
        return np.zeros(10)
    
    def simulation_step(self, states: np.ndarray, inp: np.ndarray, dt: float) -> np.ndarray:
        ''' Advances simulation by one step
            positions: Nx4 array of generation states
            returns: Nx? array of sensor positions corresponding to positions'''
        states[:,:2] = states[:,2:] * dt
        states[:,2:] = inp * dt

    def show(self, positions: np.ndarray):
        pass