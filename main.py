import numpy as np
import pyray as rl

from networks import Network, Layer
from population import GANeuralNets
from track import Track
from common import *

from math import exp, log, floor, pi

OUTPUT_NOISE = True


def rays_func(inp: np.ndarray) -> np.ndarray:
    return 20/inp


def inverse_rays_func(inp: np.ndarray) -> np.ndarray:
    return 20/inp


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


def gen_state(track: Track, pop: GANeuralNets, distribution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    population = pop.population_count
    state = np.zeros((population, 5))
    probs = distribution / np.sum(distribution)
    spawns = np.random.choice(len(track.segments), len(state), p=probs) + np.random.uniform(0, 1, len(state))
    state[:,:2] = track.evaluate_path(spawns)
    state[:,2:4] = np.random.uniform(-1, 1, (population, 2)) * 10
    state[:,4] = spawns
    return (state, spawns)


def update_state(state: np.ndarray, track: Track, pop: GANeuralNets,
                 dt: float, N_rays: int, player_input: np.ndarray=np.zeros(2)
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    rotation_angle = np.random.uniform(0, 2*pi)
    R = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)],
    ])
    if np.random.uniform(0, 1) > 0.5:
        R[0,1] *= -1
        R[1,0] *= -1

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
    inp = np.hstack((rays_func(rays), tangents))
    outp = pop.process_inputs(inp)
    if len(outp) > 0:
        outp[0] += player_input

    acc = np.zeros((len(state), 2))
    acc[:,0] = pop.extra_genomes[:,0] * outp[:,0] + pop.extra_genomes[:,1] * velocity[:,0]
    acc[:,1] = pop.extra_genomes[:,2] * outp[:,1] + pop.extra_genomes[:,3] * velocity[:,1]
    #acc = outp
    acc_norm = np.maximum(np.linalg.norm(acc, axis=1), 1e-5)  # Small offset to avoid division by 0
    acc = (acc.T * (np.tanh(acc_norm) / acc_norm).T).T * 10

    # Output corruption
    if OUTPUT_NOISE:
        acc += np.random.normal(0, 10, acc.shape)

    # Multiply by the transpose
    acc_global = np.zeros(acc.shape)
    acc_global[:,0] = R[0,0] * acc[:,0] + R[1,0] * acc[:,1]
    acc_global[:,1] = R[0,1] * acc[:,0] + R[1,1] * acc[:,1]

    state = track.simulation_step(state, acc_global, dt)
    return state, inp, outp


def training_loop_step(track: Track, pop: GANeuralNets, timesteps: int, 
                       reward_bias: float, gen_distribution: np.ndarray,
                       N_rays: int=8) -> tuple[np.ndarray, np.ndarray]:
    population = pop.population_count
    reach = np.zeros(population)
    crash_distribution = np.zeros(len(track.segments))

    state, spawns = gen_state(track, pop, gen_distribution)
    lifetime = np.zeros(population)
    time = 0
    alive = state[:,0] == state[:,0]  # initialize to true
    for timestep in range(timesteps):
        #dt = rl.get_frame_time()
        dt = 0.5
        time += dt

        pop.filter = alive
        state[alive], inp, outp, = update_state(state[alive], track, pop, dt, N_rays)
        rays = inverse_rays_func(inp[:,:N_rays])
        
        new_alive = np.logical_and(
            np.min(rays, axis=1) > 1,
            np.all(np.isfinite(state[alive]), axis=1)
        )
        died_rn = np.arange(population)[alive][np.logical_not(new_alive)]
        alive[alive] = new_alive
        for index in died_rn:
            segment_index = int(floor(state[index, 4] % len(track.segments)))
            crash_distribution[segment_index] += 1
        lifetime[alive] = time
        if not np.any(alive):
            break
        #fitness *= 1 - np.exp(-np.min(rays / 4, axis=1)/10)
    
        #fitness = -np.linalg.norm(state[:,:2], axis=1)
        reach += state[:,4] - spawns
    fitness = lifetime * (1 - reward_bias) + reach * reward_bias * 100
    fitness[alive] *= 2  # Great bonus to guys who survive to the end
    #fitness = state[:,4]
    return fitness, crash_distribution


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

    gen_distribution = np.ones(len(track.segments))

    state, spawns = gen_state(track, pop, gen_distribution)

    alive = state[:,0] == state[:,0]

    while not rl.window_should_close():
        #dt = rl.get_frame_time()
        dt = 0.0 if paused else 0.1
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            paused = not paused

        player_input = get_player_input()
        pop.filter = alive
        state[alive], inp, outp = update_state(state[alive], track, pop, dt, N_rays, player_input)
        rays = inverse_rays_func(inp[:,:N_rays])

        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(camera)
        update_camera(camera)
        
        alive[alive] = np.logical_and(
            np.min(rays, axis=1) > 1,
            np.all(np.isfinite(state[alive]), axis=1)
        )

        np_red, np_green = np.array([[1, 0, 0]]), np.array([[0, 1, 0]])
        #color_val = np.maximum(np.minimum(state[:,4:5], 1), 0)
        c = np_red * (1 - alive[:,np.newaxis]) + np_green * alive[:,np.newaxis]

        track.show_player_rays(state, N_rays)
        track.show(state, c)
        rl.end_mode_2d()
        rl.draw_fps(5, 5)
        rl.draw_text(str(state[0,4]), 5, 20, 16, FG)
        rl.end_drawing()
        
    rl.close_window()


def get_manual_net(id: str) -> np.ndarray:
    ''' Meant for simplest linear net + PD, 6 rays'''
    if id == 'track+avoidance':
        return np.array([
            -1,  -.5,    0,    0,    0,  -.5,  5,  0,  0,
             0,-.867,-.867,    0, .867, .867,  0,  5,  0,
             1,   -1,    1,   -1
        ])
    elif id == 'track':
        return np.array([
            0, 0, 0, 0, 0, 0, 5, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 5, 0,
            1,-1, 1,-1
        ])
    elif id == 'stabilize':
        return np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            1,-1, 1,-1
        ])
    return np.zeros(22)


def main():
    import matplotlib.pyplot as plt

    N_rays = 6
    pop_size = 200

    pop = GANeuralNets(pop_size, [
        Layer.linear(N_rays+2),
        #Layer.tanh(10),
        Layer.linear(2),
    ],  weight_scale=1, bias_scale=0, extra_genomes=4)
    track = Track("editing/track1.txt")

    pop.elitism = True
    pop.survivor_count = 20
    pop.mutation_rate = 0.2
    pop.mutation_stdev = 0.01
    #pop.mutation_rate = 0.001
    #pop.mutation_stdev = 0.2

    #pop.load("saved_networks/pop100_network_direct.dat")
    #pop.load("saved_networks/direct_from_scratch.dat")
    #pop.load("saved_networks/8-10-2-lifetime.dat")
    pop.genepool = np.repeat(get_manual_net('track+avoidance')[np.newaxis,:], pop_size, axis=0)
    pop.update_networks()
    
    generations = 20

    learning = np.zeros(generations)
    distribution = np.ones(len(track.segments))
    for generation in range(generations):
        fitness, distribution = training_loop_step(track, pop, 30, 1, distribution, N_rays)
        distribution += np.ones(len(track.segments)) * 100 / len(track.segments)
        pop.genetic_selection(fitness)
        learning[generation] = np.mean(fitness)
        #pop.mutation_stdev *= 0.99
        print(f"{generation:3}  {np.max(fitness):>6.2f}  {np.mean(fitness):>6.2f}")
    pop.save("saved_networks/last.dat")
    #pop.save("saved_networks/direct_from_scratch.dat")
    if generations > 0:
        plt.plot(learning)
        plt.show()
    pop2 = pop.elite_sample(20)
    visualize(track, pop, N_rays)
    #pop.save("saved_networks/speed_deep_pd.dat")
    best_candidate = pop.get_best_network()
    print(best_candidate.get_weight_matrix(0))
    print(pop.best_gene[-pop.extra_genomes_count:])



if __name__ == "__main__":
    main()