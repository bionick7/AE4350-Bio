from __future__ import annotations

import numpy as np
import pyray as rl

from networks import Network, FFNN, RBFNN, Layer
from adhdp import ADHDP, ADHDPState
from track import Track
from common import *
import cProfile

from math import exp, floor, pi

POS_MULTIPLIER = 200

class TrackState(ADHDPState):
    def __init__(self, p_track: Track, p_n_rays: int, p_internal_state: np.ndarray):
        super().__init__(p_internal_state)
        self.track = p_track
        self.n_rays = p_n_rays
    
    @classmethod
    def _dynamics(cls, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        ''' Also integrates dxdu '''
        force_multiplier = 10

        x_, y, x_dot, y_dot = x[:4]
        #acc_norm = np.maximum(np.linalg.norm(u), 1e-5)  # Small offset to avoid division by 0
        #F = force_multiplier * u * np.tanh(acc_norm) / acc_norm
        # TODO: derivative

        acceleration = force_multiplier * u * POS_MULTIPLIER

        prev_dxdt_du = x[5:].reshape(4, 2)
        dxdt_du = np.zeros((4, 2))
        dxdt_du[0:2] = prev_dxdt_du[2:4]
        dxdt_du[2:4] = force_multiplier * np.eye(2)
        dxdt = np.array([x_dot, y_dot, acceleration[0], acceleration[1], 0])

        return np.concatenate((dxdt, dxdt_du.flatten()))

    def step_forward(self, u: np.ndarray, dt: float) -> TrackState:
        next_states = np.zeros(self.internal_state.shape)
        for i in range(len(self.internal_state)):
            next_states[i] = rk4(self._dynamics, self.internal_state[i], u[i], dt)

        along_tracks, across_tracks = self.track.get_track_coordinates(next_states).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length

        return TrackState(self.track, self.n_rays, next_states)

    def get_x(self) -> np.ndarray:
        #return self.track.get_input(self.internal_state[:,:4], self.n_rays)
        x = self.internal_state[:,:4] / POS_MULTIPLIER
        x_norm = np.linalg.norm(x, axis=1)
        x[x_norm > 2] = x[x_norm > 2] / x_norm[x_norm > 2,np.newaxis]
        return x

    def get_dxdu(self) -> np.ndarray:
        dxdu = self.internal_state[:,5:].reshape(-1, 4, 2)
        dxdu /= POS_MULTIPLIER
        return dxdu

    def get_reward(self) -> np.ndarray:
        return np.linalg.norm(self.internal_state[:,:4] / POS_MULTIPLIER, axis=1)
        gamma = 0.99
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state[:,:5]).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width  # 0 ... 1
        return center_distance * (1 - gamma)
    
    def get_positions(self) -> np.ndarray:
        return self.internal_state[:,:5]
    

def rays_func(inp: np.ndarray) -> np.ndarray:
    return np.maximum(20-inp, 1e-3)


def get_player_input() -> np.ndarray:
    player_input = np.zeros(2)
    player_input[0] = rl.get_gamepad_axis_movement(0, 0) * 20
    player_input[1] = rl.get_gamepad_axis_movement(0, 1) * 20
    if rl.is_key_down(rl.KeyboardKey.KEY_LEFT):
        player_input[0] -= 20
    if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT):
        player_input[0] += 20
    if rl.is_key_down(rl.KeyboardKey.KEY_UP):
        player_input[1] -= 20
    if rl.is_key_down(rl.KeyboardKey.KEY_DOWN):
        player_input[1] += 20
    return player_input


def update_camera(camera: rl.Camera2D) -> None:
    if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
        camera.target = rl.vector2_subtract(
            camera.target, rl.vector2_scale(rl.get_mouse_delta(), 1/camera.zoom)
        )
    camera.zoom *= exp(rl.get_mouse_wheel_move() * 0.1)


def gen_state(track: Track, count: int, n_rays: int) -> TrackState:
    state_init = np.zeros((count, 13))
    spawns = np.random.choice(len(track.segments), count) + np.random.uniform(0, 1, count)
    state_init[:,:2] = track.evaluate_path(spawns)
    state_init[:,2:4] = np.random.uniform(-1, 1, (count, 2)) * 10
    state_init[:,4] = spawns
    return TrackState(track, n_rays, state_init)


def training_loop(track: Track, adhdp: ADHDP, N_rays: int=8) -> None:
    states = gen_state(track, 20, N_rays)
    for _ in range(300):
        states = adhdp.step_and_learn(states, 0.1)


def visualize(track: Track, adhdp: ADHDP, N_rays: int=8):
    # Raylib initialisation
    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        #| rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    #rl.set_target_fps(60)
    rl.init_window(1000, 600, ":3")
    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)
    paused = True

    # Generate states

    states = gen_state(track, 20, N_rays)

    while not rl.window_should_close():
        dt = 0.1
        dt = rl.get_frame_time()

        # User Input
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            paused = not paused
        #if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
        #    np.savetxt("saved_networks/critic_weights.dat", critic.weights)
        if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            states = gen_state(track, 20, N_rays)
            
        # run adhdp
        if not paused or rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
            states = adhdp.step_and_learn(states, dt)
        
        if np.average(states.get_reward()) > 100:
            break
        if np.average(states.get_reward()) < 1e-3:
            break

        # Drawing
        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(camera)
        update_camera(camera)
        
        xx = states.get_positions()
        track.show_player_rays(xx, N_rays)
        track.show(xx)
        rl.end_mode_2d()
        rl.draw_fps(5, 5)
        rl.draw_text(str(adhdp.J_prev[0]), 5, 20, 16, FG)
        rl.end_drawing()
        
    rl.close_window()

def main():
    import matplotlib.pyplot as plt

    population = 20
    track = Track("editing/track1.txt")

    #actor = FFNN([
    #    Layer.linear(4),
    #    Layer.tanh(4),
    #    Layer.tanh(2),
    #], (-1, 1), (0, 0))
    #actor.load_weights_from("saved_networks/track/actor_start.dat")
    
    actor = FFNN([
        Layer.linear(4),
        Layer.linear(2),
    ], (-1, 1), (0, 0))
    actor.set_weight_matrix(0, -np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0]
    ]))
    
    critic = FFNN([
        Layer.linear(6),
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(1),
    ], (-1, 1), (-1, 1))
    critic.load_weights_from("saved_networks/track/critic_start.dat")

    adhdp = ADHDP(actor, critic, population)
    adhdp.gamma = 0.99

    adhdp.train_actor = True
    adhdp.train_critic = False

    N_rays = 6
    visualize(track, adhdp, N_rays)

if __name__ == "__main__":
    main()