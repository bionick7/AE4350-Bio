import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from track_states import TrackStateRot, gen_state
from networks import Network, FFNN, RBFNN, Layer, check_io_gradients
from adhdp import ADHDP, ADHDPState
from track import Track, TrackSegmentArc
from common import *

from math import exp

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


def visualize(track: Track, adhdp: ADHDP, population: int):
    # Raylib initialisation
    rl.set_config_flags(0
        | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
        #| rl.ConfigFlags.FLAG_VSYNC_HINT
    )
    #rl.set_target_fps(60)
    rl.init_window(1000, 600, ":3")  # TODO: Maybe change
    camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)
    paused = True

    fig, ax = plt.subplots()
    learning = []

    outer_loop = True
    while outer_loop:
        # Generate states
        states = TrackStateRot(track, gen_state(track, population, True))
        time = 0
        while True:
            dt = DELTA_TIME
            #dt = rl.get_frame_time()
            
            # run adhdp
            if not paused or rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                
                #if adhdp.gamma < 0.99:
                #    adhdp.gamma = 1 - (1-adhdp.gamma) * 0.79

                time += dt

                states: TrackStateRot = adhdp.step_and_learn(states, dt)
            
                if np.average(states.get_reward(adhdp)) > 100:
                    break
                #if np.median(states.get_reward(adhdp)) < 1e-2:
                #    break
                #if time > 10:
                #    break

                #print(np.average(adhdp.error[1]))
                learning.append(np.average(adhdp.error, axis=1))


            # Drawing
            rl.begin_drawing()
            rl.clear_background(BG)
            rl.begin_mode_2d(camera)
            update_camera(camera) 
            
            xx = states.get_positions()
            track.show(xx)
            track.show_player_rays(states.internal_state, 4)

            rl.end_mode_2d()
            rl.draw_fps(5, 5)
            rl.draw_text(str(adhdp.error[0,0]), 5, 20, 16, FG)
            rl.draw_text(str(states.get_reward(adhdp)[0]), 5, 40, 16, FG)
            rl.end_drawing()

            #print(states.internal_state[0])
            
            # User Input
            if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
                paused = not paused
            if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
                adhdp.actor.save_weights_to("norays_rot/actor_prepped.dat")
                adhdp.critic.save_weights_to("norays_rot/critic_rbf.dat")
                #adhdp.plant.save_weights_to("norays_vel/plant.dat")
                pass

            if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                break
                
            if rl.window_should_close():
                outer_loop = False
                break

            #if len(learning) > 0:
            #    break
        
        ax.clear()
        learning_array = np.array(learning)
        ax.plot(learning_array[:,0], label="actor")
        ax.plot(learning_array[:,1], label="critic")
        ax.plot(learning_array[:,2], label="plant")
        #ax.set_ylim((0, 0.01))
        ax.grid()
        ax.legend()
        plt.pause(1e-10)
        
    rl.close_window()
    #plt.plot(learning_c, label="actor")
    #plt.plot(learning_a, label="critic")
    #plt.legend()
    #plt.ylim((0, 3))


def generate_networks() -> tuple[Network, Network, Network]:
    
    actor = RBFNN.grid_spaced(1,
        np.linspace(-1, 1, 30), 
        np.linspace(-1, 1, 5))

    #actor = FFNN([
    #    Layer.linear(2),
    #    Layer.tanh(20),
    #    Layer.tanh(2),
    #], (-1, 1), (-1, 1))

    #critic = RBFNN.grid_spaced(1, 
    #    np.linspace(-1, 1, 6), 
    #    np.linspace(-1, 1, 4), 
    #    np.linspace(-1, 1, 5))

    critic = FFNN([
        Layer.linear(3),
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(1),
    ], (-1, 1), (-1, 1))

    plant = FFNN([
        Layer.linear(3),
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(2),
    ])

    actor.load_weights_from("norays_rot/actor_prepped.dat")
    critic.load_weights_from("norays_rot/critic.dat")
    #plant.load_weights_from("norays_vel/plant.dat")

    return actor, critic, plant


def main():
    import matplotlib.pyplot as plt

    population = 30
    track = Track("editing/track1.txt")
    
    adhdp = ADHDP(*generate_networks(), population)
    adhdp.u_offsets = np.random.normal(0, 0.0, (population, 1))
    adhdp.gamma = 0.99

    adhdp.train_actor = False
    adhdp.train_critic = True
    adhdp.train_plant = False
    adhdp.train_actor_on_initial = True

    adhdp.use_plant = False
    adhdp.use_actor = True
    adhdp.actor_learning_rate = 1e-2
    adhdp.critic_learning_rate = 1e-2
    adhdp.plant_learning_rate = 1e-3
    #check_io_gradients(adhdp.critic)

    #test_states = TrackState(track, gen_state(track, population, False))
    #test_states.check_dsdu()

    visualize(track, adhdp, population)
    adhdp.plot_critic_gradient(0,1, 0,1)
    adhdp.plot_actor_critic(0,1)
    #adhdp.plot_actor_critic(2,3)
    plt.show()


if __name__ == "__main__":
    main()