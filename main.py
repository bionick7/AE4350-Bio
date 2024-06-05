import numpy as np
import pyray as rl

from track import Track
from common import *


def control(inp: np.ndarray) -> np.ndarray:
    N = len(inp)
    # TODO
    return np.random.uniform(-30, 30, (N, 2))
    #return np.zeros((N, 2))


def main():
    track = Track("editing/track1.dat")
    state = np.zeros((10, 4))
    state[:,0] = 500
    state[:,1] = 100

    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        | rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    rl.set_target_fps(60)
    rl.init_window(1000, 600, "Window")

    while not rl.window_should_close():
        dt = rl.get_frame_time()

        rl.begin_drawing()
        rl.clear_background(BG)

        inp = track.get_input(state)
        outp = control(inp)
        state = track.simulation_step(state, outp, dt*10)

        track.show(state)
        
        rl.end_drawing()
        rl.draw_fps(5, 5)
    rl.close_window()


if __name__ == "__main__":
    main()