import numpy as np
import pyray as rl

from networks import Network, Layer
from population import GANeuralNets
from track import Track
from common import *

from math import exp

def update_camera(camera: rl.Camera2D) -> None:
    if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
        camera.target = rl.vector2_subtract(
            camera.target, rl.vector2_scale(rl.get_mouse_delta(), 1/camera.zoom)
        )
    camera.zoom *= exp(rl.get_mouse_wheel_move() * 0.1)


def training_loop(track: Track, ga_population: GANeuralNets, generations: int, 
                  timesteps: int, reward_bias: float, N_rays: int=8) -> None:
    #track = Track()
    population = ga_population.population_count

    #print("-"*10)
    for generation in range(generations):
        #print("#", end="", flush=True)

        state = np.zeros((population, 5))
        state[:,:2] = track.starting_point
        state[:,2:4] = np.random.uniform(-1, 1, (population, 2)) * 10
        state[:,4] = 0

        time = 0
        lifetime = np.zeros(population)
        speed = np.zeros(population)
        alive = state[:,0] == state[:,0]  # initialize to true
        for timestep in range(timesteps):
            #dt = rl.get_frame_time()
            dt = 0.5
            time += dt

            rays = track.get_input(state[alive], N_rays)
            inp = np.hstack((rays, state[alive,2:4]))
            outp = ga_population.process_inputs(inp)
            output_norm = np.linalg.norm(outp, axis=1)
            outp = (outp.T * (np.tanh(output_norm) / output_norm).T).T * 30

            state[alive] = track.simulation_step(state[alive], outp, dt)
            alive[alive] = np.logical_and(
                np.min(rays, axis=1) > 1,
                np.all(np.isfinite(state[alive]), axis=1)
            )
            lifetime[alive] = time
            speed[alive] = state[alive,4] / time
            if not np.any(alive):
                break
            #fitness *= 1 - np.exp(-np.min(rays / 4, axis=1)/10)
        
        #fitness = -np.linalg.norm(state[:,:2], axis=1)
        fitness = (lifetime / max(lifetime)) * (1 - reward_bias) + (speed / max(speed)) * reward_bias
        fitness = lifetime
        ga_population.genetic_selection(fitness)
        print(max(lifetime), np.mean(lifetime), max(speed), np.mean(speed))


def visualize(track: Track, pop: GANeuralNets, N_rays: int=8):
    #track = Track()

    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        | rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    rl.set_target_fps(60)
    rl.init_window(1000, 600, "Genetic Algorythm Visualizer")
    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)

    state = np.zeros((pop.population_count, 5))
    state[:,:2] = track.starting_point
    state[:,2:4] = np.random.uniform(-1, 1, (pop.population_count, 2)) * 1
    state[:,4] = 0

    paused = True
    fitness = np.ones(pop.population_count) * 100

    while not rl.window_should_close():
        #dt = rl.get_frame_time()
        dt = 0.0 if paused else 0.1
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            paused = not paused

        rays = track.get_input(state, N_rays)
        inp = np.hstack((rays, state[:,2:4]))
        outp = pop.process_inputs(inp)
        output_norm = np.linalg.norm(outp, axis=1)
        outp = (outp.T * (np.tanh(output_norm) / output_norm).T).T * 30
        state = track.simulation_step(state, outp, dt)

        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(camera)
        update_camera(camera)

        # 'Player' controls speed directly
        state[0,2] = rl.get_gamepad_axis_movement(0, 0) * 20
        state[0,3] = rl.get_gamepad_axis_movement(0, 1) * 20
        if rl.is_key_down(rl.KeyboardKey.KEY_LEFT):
            state[0,2] -= 20
        if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT):
            state[0,2] += 20
        if rl.is_key_down(rl.KeyboardKey.KEY_UP):
            state[0,3] -= 20
        if rl.is_key_down(rl.KeyboardKey.KEY_DOWN):
            state[0,3] += 20
            
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
    N_rays = 6

    pop = GANeuralNets(50, [
        Layer.linear(N_rays+2),
        Layer.tanh(1),
        Layer.tanh(2),
    ], p_scale=0.1)
    track0 = Track()
    track1 = Track("editing/track1.txt")

    pop.survivor_count = 20
    training_loop(track0, pop, 10, 10000, 0, N_rays)
    visualize(track0, pop, N_rays)
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