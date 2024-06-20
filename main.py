import numpy as np
import pyray as rl

from track import Track
from common import *

from math import exp

def control(inp: np.ndarray) -> np.ndarray:
    N = len(inp)
    # TODO
    outp = np.random.uniform(-30, 30, (N, 2))
    # 'Player' controls first one
    #outp[0,0] = rl.get_gamepad_axis_movement(0, 0) * 20
    #outp[0,1] = rl.get_gamepad_axis_movement(0, 1) * 20

    return outp
    #return np.zeros((N, 2))


def update_camera(camera: rl.Camera2D) -> None:
    if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
        camera.target = rl.vector2_subtract(
            camera.target, rl.vector2_scale(rl.get_mouse_delta(), 1/camera.zoom)
        )
    camera.zoom *= exp(rl.get_mouse_wheel_move() * 0.1)

def main():
    track = Track("editing/track1.txt")
    #track = Track()
    state = np.zeros((10, 5))
    state[:,0] =  0
    state[:,1] = -100
    state[:,4] = 0

    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        | rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    rl.set_target_fps(60)
    rl.init_window(1000, 600, "Window")

    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)

    while not rl.window_should_close():
        dt = rl.get_frame_time()
        dt = 0.1

        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(camera)

        update_camera(camera)

        inp = track.get_input(state)
        outp = control(inp)
        state = track.simulation_step(state, outp, dt)
        
        # 'Player' controls speed directly
        #state[0,2] = rl.get_gamepad_axis_movement(0, 0) * 20
        #state[0,3] = rl.get_gamepad_axis_movement(0, 1) * 20

        track.show(state)
        #track.show_player_rays(state)
        
        rl.end_mode_2d()
        rl.end_drawing()
        rl.draw_fps(5, 5)
    rl.close_window()


if __name__ == "__main__":
    main()