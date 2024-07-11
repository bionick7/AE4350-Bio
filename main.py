import numpy as np
import pyray as rl

from networks import Network, FFNN, RBFNN, Layer
from population import GANeuralNets
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
                 dt: float, N_rays: int, player_input: np.ndarray=np.zeros(2)
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
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


def critic_training(critic: Network, inp: np.ndarray, outp: np.ndarray, 
                    r: float, J_prev: float, training: dict) -> tuple[float, float]:

    critic_inp = np.concatenate((inp, outp))
    critic_outp, dwdy = critic.get_gradients(critic_inp)
    J: float = critic_outp.flatten()[0]
    e: float = J_prev - (training['gamma'] * J + r)

    mu = 0.2
    #delta_weights = -np.linalg.solve(dwdy.T@dwdy - mu * np.eye(dwdy.shape[1]), dwdy.T*e).flatten()
    delta_weights = -dwdy.flatten()*e * training['alpha']
    critic.update_weights(delta_weights)
    return J, .5*e**2

def init_training_details(state: np.ndarray) -> dict:
    return {
        'gamma': 0.5,
        'alpha': 1e-4,
        'J': np.zeros(len(state))
    }

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


def training_loop_step(track: Track, critic: Network, N_rays: int=8) -> None:
    population = 20
    state, spawns = gen_state(track, population)
    training = init_training_details(state)
    for _ in range(300):
        dt = 0.1
        state = update_state_and_train(track, state, spawns, critic, training, dt, N_rays)


def visualize(track: Track, critic: Network, N_rays: int=8):
    #track = Track()

    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        | rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    rl.set_target_fps(60)
    rl.init_window(1000, 600, "Genetic Algorythm Visualizer")
    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)
    paused = True

    state, spawns = gen_state(track, 20)
    training = init_training_details(state)

    while not rl.window_should_close():
        #dt = rl.get_frame_time()
        dt = 0.0 if paused else 0.1
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            paused = not paused

        if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
            np.savetxt("saved_networks/critic_weights.dat", critic.weights)
            
        if rl.is_key_pressed(rl.KeyboardKey.KEY_G):
            training['gamma'] += 0.1

        player_input = get_player_input()
        if not paused:
            state = update_state_and_train(track, state, spawns, critic, training, dt, N_rays)

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

    N_rays = 6
    weight_range = 5
    critic = FFNN([
        Layer.linear(12),
        Layer.tanh(10),
        #Layer.tanh(20),
        Layer.linear(1)
    ], w_clamp=(-weight_range,weight_range), b_clamp=(-weight_range,weight_range))
    #critic.weights = np.loadtxt("saved_networks/critic_weights.dat")

    track = Track("editing/track2.txt")
    visualize(track, critic, N_rays)
    print(critic.get_weight_matrix(0)[:,-1])
    print(critic.get_weight_matrix(1)[:,-1])
    print(critic.get_weight_matrix(2)[:,-1])
    print(critic.get_weight_matrix(2))
    #training_loop_step(track, critic, N_rays)


if __name__ == "__main__":
    main()