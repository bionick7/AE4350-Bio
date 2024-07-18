import numpy as np
import pyray as rl

from networks import Network, FFNN, RBFNN, Layer
from rl_test import acceleration_dynamics_2d, rk4
from adhdp import ADHDP
from track import Track
from common import *

from math import exp, log, floor, pi

OUTPUT_NOISE = True


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


def gen_state(track: Track, count: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((count, 5))
    spawns = np.random.choice(len(track.segments), len(state)) + np.random.uniform(0, 1, len(state))
    state[:,:2] = track.evaluate_path(spawns)
    state[:,2:4] = np.random.uniform(-1, 1, (count, 2)) * 10
    state[:,4] = spawns
    return (state, spawns)


def classical_control(inp: np.ndarray, N_rays: int) -> np.ndarray:
    speed = 30
    K = 40
    manual_nn = K * np.array([
        [-1,  -.5,   .5, 1,   .5,  -.5],
        [ 0,-.867,-.867, 0, .867, .867],
    ])

    outp = (manual_nn@inp[:,:N_rays].T).T
    outp += inp[:,N_rays+0:N_rays+2] * speed
    acc = outp - inp[:,N_rays+2:]
    return acc


def update_state(state: np.ndarray, track: Track,
                 dt: float, N_rays: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    rotation_angle = np.random.uniform(0, 2*pi)
    R = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)],
    ])
    if np.random.uniform(0, 1) > 0.5:
        # mirror
        R[0,1] *= -1

    #R = np.eye(2)
    rays = track.get_input(state, N_rays, R)

    tangents_global = track.get_path_dir(state)
    tangents = np.zeros(tangents_global.shape)
    tangents[:,0] = R[0,0] * tangents_global[:,0] + R[0,1] * tangents_global[:,1]
    tangents[:,1] = R[1,0] * tangents_global[:,0] + R[1,1] * tangents_global[:,1]
    
    velocity = np.zeros(state[:,2:4].shape)
    velocity[:,0] = R[0,0] * state[:,2] + R[0,1] * state[:,3]
    velocity[:,1] = R[1,0] * state[:,2] + R[1,1] * state[:,3]

    #inp = np.hstack((1/rays, state[:,2:4], tangents))
    inp = np.hstack((rays_func(rays), tangents, velocity))

    acc = classical_control(inp, N_rays)

    #normalize acceleration
    acc_norm = np.maximum(np.linalg.norm(acc, axis=1), 1e-5)  # Small offset to avoid division by 0
    acc_2 = (acc.T * (np.tanh(acc_norm) / acc_norm).T).T * 10

    # Output corruption
    if OUTPUT_NOISE:
        acc_2 += np.random.normal(0, 10, acc_2.shape)

    # Multiply by the transpose
    acc_global = np.zeros(acc_2.shape)
    acc_global[:,0] = R[0,0] * acc_2[:,0] + R[1,0] * acc_2[:,1]
    acc_global[:,1] = R[0,1] * acc_2[:,0] + R[1,1] * acc_2[:,1]

    state = track.simulation_step(state, acc_global, dt)
    return state, acc, inp


def update_state_and_train(track: Track, state: np.ndarray, spawns: np.ndarray, critic: Network, training_data: dict,
                           dt: float, N_rays: int=8) -> np.ndarray:
    ''' Relies on mutability of training_data '''
    population = len(state)
    errors = np.zeros(population)
    state, acc, inp = update_state(state, track, dt, N_rays)
    
    reach = state[:,4] - spawns
    center_distance = 2*np.abs(track.get_track_coordinates(state)[:,1])/track.track_width  # -1 ... 1
    fitness = center_distance
    for i in range(population):
        J = training_data['J'][i]
        training_data['J'][i], E = critic_training(critic, inp[i], acc[i], fitness[i], J, training_data)
        errors[i] = E / population

    print(f"{training_data['gamma']:.2f} {np.mean(errors):8.4f} {np.std(errors):8.4f}"
          f"  {fitness[0]:6.4f}, {training_data['J'][0]:6.4f}")
    return state


def state_derivative(state: np.ndarray, u: np.ndarray, track: Track, dt: float) -> np.ndarray:
    x = state[:4]
    dxdu1 = state[5:9]
    dxdu2 = state[10:]
    res = x.copy()
    res[:4] = rk4(acceleration_dynamics_2d, x[:4], u, dt)
    segment_index = int(floor(state[4] % len(track.segments)))
    along_track, across_track = track.segments[segment_index].get_track_coordinates(x[np.newaxis,:2])[0]
    res[4] = along_track
    return res


def reward(x: np.ndarray, track: Track) -> float:
    center_distance = 2*np.abs(track.get_track_coordinates(x[np.newaxis,:])[0,1])/track.track_width  # -1 ... 1
    return center_distance*center_distance


def training_loop(track: Track, adhdp: ADHDP, N_rays: int=8) -> None:
    population = 20
    state, spawns = gen_state(track, population)
    for _ in range(300):
        dt = 0.1
        state[:4] = adhdp.step_and_learn(state, lambda x: x[:4],
                                        lambda x, u: state_derivative(x, u, track, dt), 
                                        lambda x: reward(x, track))


def visualize(track: Track, adhdp: ADHDP, N_rays: int=8):
    #track = Track()

    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        | rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    rl.set_target_fps(60)
    rl.init_window(1000, 600, "Genetic Algorithm Visualizer")
    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)
    paused = True

    state, spawns = gen_state(track, 20)

    while not rl.window_should_close():
        #dt = rl.get_frame_time()
        dt = 0.0 if paused else 0.1
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            paused = not paused

        if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
            #np.savetxt("saved_networks/critic_weights.dat", critic.weights)
            pass
            
        training_loop(track, adhdp, N_rays)

        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(camera)
        update_camera(camera)
        
        track.show_player_rays(state, N_rays)
        track.show(state)
        rl.end_mode_2d()
        rl.draw_fps(5, 5)
        rl.draw_text(str(state[0,4]), 5, 20, 16, FG)
        rl.end_drawing()
        
    rl.close_window()

def main():
    import matplotlib.pyplot as plt

    population = 20
    track = Track("editing/track2.txt")

    actor = FFNN([
        Layer.linear(4),
        Layer.tanh(4),
        Layer.linear(2),
    ], (-1, 1), (0, 0))
    actor.load_weights_from("saved_networks/acc2d/actor_trained_p99.dat")
    critic = FFNN([
        Layer.linear(6),
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(1),
    ], (-1, 1), (-1, 1))
    critic.load_weights_from("saved_networks/acc2d/critic_trained_p99.dat")

    adhdp = ADHDP(actor, critic, population)

    N_rays = 6
    training_loop(track, adhdp, N_rays)


if __name__ == "__main__":
    main()