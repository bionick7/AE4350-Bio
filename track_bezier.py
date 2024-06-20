import numpy as np
import pyray as rl

from common import *
from collision_detection import get_bezier_coeffs, bezier_distance

class TrackBezie:
    def __init__(self, load_path: str|None=None) -> None:
        self.track_width = 40
        if load_path == None:
            # Simple circle spline
            spline_delta = 110.426
            splines_array = np.array([
                [0, -200], [spline_delta, -200], 
                [200, -spline_delta],  [200, 0], [200, spline_delta],
                [spline_delta, 200],  [0, 200], [-spline_delta, 200],
                [-200, spline_delta],  [-200, 0], [-200, -spline_delta],
                [-spline_delta, -200], [0, -200], 
            ])
            splines_array[:,0] += 500
            splines_array[:,1] += 300
            spline_count = len(self.splines_array)//3
            self.splines = np.zeros((spline_count, 4, 2))
            for i in range(0, spline_count):
                self.splines[i,:] = self.splines_array[i*3:i*3+4]
        else:
            self.splines = np.loadtxt(load_path).reshape(-1, 4, 2)
            spline_count = len(self.splines)

        self.spline_coeffs = np.zeros((spline_count, 4, 2))

        for i in range(0, spline_count):
            self.spline_coeffs[i] = get_bezier_coeffs(self.splines[i])

        print(self.splines)
        print(self.spline_coeffs)

    def get_input(self, states: np.ndarray) -> np.ndarray:
        ''' Returns sensor readings evaluated at a certain point 
            states: Nx4 array of generation states
            returns: Nx? array of sensor positions corresponding to positions'''
        N = len(states)
        # TODO
        return np.zeros((N, 10))
    
    def simulation_step(self, states: np.ndarray, outp: np.ndarray, dt: float) -> np.ndarray:
        ''' Advances simulation by one step
            states: Nx4 array of generation states
            outp: Nx2 array with accelerations
            returns: Nx? array of sensor positions corresponding to positions'''
        states[:,:2] += states[:,2:] * dt
        states[:,2:] += outp * dt

        for state_index, state in enumerate(states):
            distances = np.zeros((len(self.splines), 3))
            for spline_index, coeffs in enumerate(self.spline_coeffs):
                distances[spline_index,1:], distances[spline_index,0] = bezier_distance(coeffs.reshape(1, -1, 2) , state[:2])
            index = np.argmin(distances[:,0])
            if distances[index, 0] > self.track_width//2:
                dir = (distances[index, 1:] - state[:2]) / distances[index, 0]
                states[state_index,:2] = distances[index, 1:] - dir * (self.track_width//2 - 0.001)
                states[state_index,2:] = dir*2

        # Might bump into stuff
        return states

    def show(self, state: np.ndarray):
        #pts = [[100, 100], [100, 200], [500, 30], [400, 200]]
        for spline in self.splines:
            rl.draw_spline_segment_bezier_cubic(
                spline[0].tolist(), 
                spline[1].tolist(),
                spline[2].tolist(),
                spline[3].tolist(),
                self.track_width, FG)
        #for pt in pts:
        #    rl.draw_circle(pt[0], pt[1], 10, HIGHLIGHT)

        for pos in state[:,:2]:
            rl.draw_circle(int(pos[0]), int(pos[1]), 4.0, HIGHLIGHT)
