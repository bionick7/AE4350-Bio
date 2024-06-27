import numpy as np
import pyray as rl

from networks import Network, Layer
from population import GANeuralNets
from track import Track
from common import *

from math import exp, log, floor

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


def gen_state(track: Track, pop: GANeuralNets) -> tuple[np.ndarray, np.ndarray]:
    population = pop.population_count
    state = np.zeros((population, 5))
    spawns = np.random.uniform(0, len(track.segments), len(state))
    state[:,:2] = track.evaluate_path(spawns)
    state[:,2:4] = np.random.uniform(-1, 1, (population, 2)) * 10
    state[:,4] = spawns
    return (state, spawns)


def update_state(state: np.ndarray, track: Track, pop: GANeuralNets,
                 dt: float, N_rays: int, player_input: np.ndarray=np.zeros(2)
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = track.get_input(state, N_rays)
    tangents = track.get_path_dir(state)
    inp = np.hstack((rays / 20, tangents))
    outp = pop.process_inputs(inp)
    outp[0] += player_input

    K = 1
    acc = K * (outp * 10 - state[:,2:4])
    acc_norm = np.maximum(np.linalg.norm(acc, axis=1), 1e-5)  # Small offset to avoid division by 0
    acc = (acc.T * (np.tanh(acc_norm) / acc_norm).T).T * 30

    # Output corruption
    #acc += np.random.normal(0, 10, acc.shape)

    state = track.simulation_step(state, acc, dt)
    return state, inp, outp


def training_loop_step(track: Track, pop: GANeuralNets, timesteps: int, 
                       reward_bias: float, critic: Network, N_rays: int=8) -> np.ndarray:
    population = pop.population_count
    #print("#", end="", flush=True)

    cumulative_lifetime = np.zeros(population)
    reach = np.zeros(population)

    J_prev = np.zeros(population)
    gamma = 0.2

    cum_critic_error = 0
    critic_evals = 0

    for test in range(1):
        state, spawns = gen_state(track, pop)
        lifetime = np.zeros(population)
        time = 0
        alive = state[:,0] == state[:,0]  # initialize to true
        for timestep in range(timesteps):
            #dt = rl.get_frame_time()
            dt = 0.5
            time += dt

            state[alive], inp, outp, = update_state(state[alive], track, pop, dt, N_rays)
            rays = inp[:,:N_rays] * 20
            
            # Reintroduce critic
            for i in range(len(inp)):
                reward = state[alive][i,4] - spawns[alive][i]
                outp_critic, grad_critic = critic.get_gradients(np.concatenate((inp[i], outp[i])))
                J = outp_critic[0,0]
                e_c = np.array([[J_prev[alive][i] - (gamma * J - reward)]])
                E_c = .5 * (e_c.T@e_c)**2
                cum_critic_error += E_c
                critic_evals += 1
            
                # update critic weights
                critic_weight_delta = (grad_critic.T @ e_c).flatten() * 0.01
                #critic_weight_delta = np.linalg.solve(
                #    grad_critic.T@grad_critic - 0.1 * np.eye(grad_critic.shape[1]), 
                #    grad_critic.T@e_c
                #).flatten()
                critic.update_weights(critic_weight_delta)
            
                J_prev[alive][i] = J
        
            alive[alive] = np.logical_and(
                np.min(rays, axis=1) > 1,
                np.all(np.isfinite(state[alive]), axis=1)
            )
            lifetime[alive] = time
            if not np.any(alive):
                break
            #fitness *= 1 - np.exp(-np.min(rays / 4, axis=1)/10)
    
        #fitness = -np.linalg.norm(state[:,:2], axis=1)
        cumulative_lifetime += lifetime
        reach += state[:,4] - spawns
    fitness = cumulative_lifetime * (1 - reward_bias) + reach * reward_bias * 100
    #fitness = state[:,4]
    print(cum_critic_error / critic_evals)
    pop.genetic_selection(fitness)
    return fitness


def visualize(track: Track, pop: GANeuralNets, N_rays: int=8):
    #track = Track()

    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        | rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    rl.set_target_fps(60)
    rl.init_window(1000, 600, "Genetic Algorythm Visualizer")
    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)
    paused = True

    state, spawns = gen_state(track, pop)

    fitness = np.ones(pop.population_count) * 100

    while not rl.window_should_close():
        #dt = rl.get_frame_time()
        dt = 0.0 if paused else 0.1
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            paused = not paused

        player_input = get_player_input()
        state, inp, outp = update_state(state, track, pop, dt, N_rays, player_input)
        rays = inp[:,:N_rays] * 20
        #state[0,2:4] = player_input * 5

        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(camera)
        update_camera(camera)

        # 'Player' controls speed directly
            
        fitness *= 1 - np.exp(-np.min(rays / 4, axis=1))
        np_red, np_green = np.array([[1, 0, 0]]), np.array([[0, 1, 0]])
        color_val = np.maximum(np.minimum(state[:,4:5], 1), 0)
        c = np_red * (1 - color_val) + np_green * color_val

        track.show_player_rays(state, N_rays)
        track.show(state, c)
        rl.end_mode_2d()
        rl.draw_fps(5, 5)
        rl.draw_text(str(state[0,4]), 5, 20, 16, FG)
        rl.end_drawing()
        
    rl.close_window()


def main():
    import matplotlib.pyplot as plt

    N_rays = 6

    pop = GANeuralNets(50, [
        Layer.linear(N_rays+2),
        Layer.tanh(10),
        Layer.linear(2),
    ], p_scale=1)
    track0 = Track()
    track1 = Track("editing/track2.txt")

    pop.elitism = True
    pop.survivor_count = 20
    pop.mutation_rate = 0.2
    pop.mutation_stdev = 0.01

    pop.load("saved_networks/speed_deep_pd.dat")

    generations = 50

    learning = np.zeros(generations)

    critic = Network([
        Layer.linear(N_rays+4),
        Layer.tanh(10),
        Layer.linear(1),
    ])

    for generation in range(generations):
        fitness = training_loop_step(track1, pop, 300, 0, critic, N_rays)
        learning[generation] = np.mean(fitness)
        #pop.mutation_stdev *= 0.99
        print(f"{generation:3}  {max(fitness):>6.2f}  {np.mean(fitness):>6.2f}")
    plt.plot(learning)
    plt.show()
    visualize(track1, pop, N_rays)
    #pop.save("saved_networks/speed_deep_pd.dat")
    #training_loop(track1, pop, 10, 10000, 0, N_rays)
    #training_loop(pop, 10, 10000, 0.5, N_rays)
    #training_loop(pop, 6, 50, N_rays)
    best_candidate = pop.get_best_network()
    print(best_candidate.get_weight_matrix(0))
    #visualize(track1, pop, N_rays)

    #pop = GANeuralNets(10, [
    #    Layer.linear(N_rays+2),
    #    Layer.linear(2),
    #], scale=0.1)
    #visualize(pop, N_rays)


if __name__ == "__main__":
    main()