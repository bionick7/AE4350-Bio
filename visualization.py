import pyray as rl
import numpy as np
import matplotlib.pyplot as plt
from math import exp

from track import Track
from adhdp import ADHDP
from track_states import *

def update_camera(camera: rl.Camera2D) -> None:
    if rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT):
        camera.target = rl.vector2_subtract(
            camera.target, rl.vector2_scale(rl.get_mouse_delta(), 1/camera.zoom)
        )
    camera.zoom *= exp(rl.get_mouse_wheel_move() * 0.1)


class Visualization:
    def __init__(self, p_track: Track, p_population: int):
        self.track = p_track
        self.population = p_population
        
        self.camera = rl.Camera2D(rl.Vector2(500, 300), rl.Vector2(0, 0), 0, 1)
        self.fig, self.ax = plt.subplots()

        self.paused = True
        self.is_running = True
        self.break_epoch = False
        self.save_requested = False

        # Raylib initialisation
        rl.set_config_flags(0
            | rl.ConfigFlags.FLAG_WINDOW_RESIZABLE 
            #| rl.ConfigFlags.FLAG_VSYNC_HINT
        )
        #rl.set_target_fps(60)
        rl.init_window(1000, 600, "Racetrack visualize")

    def show(self, states: TrackState, u: np.ndarray):
        # Drawing
        rl.begin_drawing()
        rl.clear_background(BG)
        rl.begin_mode_2d(self.camera)
        update_camera(self.camera) 
        
        xx = states.get_positions()
        self.track.show(xx, u)
        self.track.show_player_rays(states.internal_state, 4)

        rl.end_mode_2d()
        rl.draw_fps(5, 5)
        #rl.draw_text(str(self.adhdp.error[0,0]), 5, 20, 16, FG)
        #rl.draw_text(str(states.get_reward(self.adhdp)[0]), 5, 40, 16, FG)
        rl.end_drawing()

        #print(states.internal_state[0])
        
        # User Input
        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
            self.paused = not self.paused

        if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
            self.save_requested = True

        if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            self.break_epoch = True
            
        if rl.window_should_close():
            self.is_running = False
            self.break_epoch = True

    def update_graph(self, graphs, labels: list[str]):
        self.ax.clear()
        graph_array = np.array(graphs)
        if len(graph_array) < 0:
            return
        
        for i, label in enumerate(labels):
            self.ax.plot(graph_array[:,i], label=label)
        #ax.plot(learning_array[:,3], label="reward")
        #ax.set_ylim((0, 0.01))
        self.ax.grid()
        self.ax.legend()
        plt.pause(1e-10)

    def end(self):
        pass


def visualize_adhdp(track: Track, adhdp: ADHDP, population: int, constrain_weights: bool):

    vis = Visualization(track, population)

    learning = []

    win_condition = 1
    if constrain_weights:
        centers_track_positions = (adhdp.actor.centers[:,0] + 1) / 2 * len(track.segments)
        adhdp.actor_weight_mask = np.append(np.exp(-np.square(centers_track_positions - win_condition)), 0)

    best_actor_weights = adhdp.actor.weights
    best_reach = 2

    while vis.is_running:
        # Generate states
        states = TrackStateRot(track, gen_state(track, population, True))
        states.win_condition = win_condition
        time = 0

        vis.break_epoch = False
        while not vis.break_epoch:
            dt = DELTA_TIME
            #dt = rl.get_frame_time()
            
            # run adhdp
            if not vis.paused or rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                time += dt
                states: TrackStateRot = adhdp.step_and_learn(states, dt)
            
                if time > 20:
                    break

                if np.all(states.reset):
                    break

                #print(np.average(adhdp.error[1]))
                #avg_performance = np.average(adhdp.error, axis=1)

            vis.show(states, adhdp.u)
            if vis.save_requested:
                adhdp.save_networks()
        
        new_reach = np.average(states.internal_state[:,4])
        learning.append([new_reach])
        vis.update_graph(learning, ["progress"])

        # Decide if to revert
        if adhdp.train_actor and adhdp.train_critic:
            if new_reach > best_reach:
                best_reach = new_reach 
                best_actor_weights = adhdp.actor.weights
                #old_critic_weights = adhdp.critic.weights
            else:
                centers_track_positions = (adhdp.actor.centers[:,0] + 1) / 2 * len(track.segments)
                actor_weight_mask = np.append(np.exp(-np.square(centers_track_positions - win_condition)), 0)
                adhdp.actor.weights = best_actor_weights# + np.random.normal(0, 0.01, best_actor_weights.shape) * actor_weight_mask
                
        if np.count_nonzero(states.win_mask) > len(states) / 2:
            win_condition = min(win_condition + 0.1, len(track.segments))
            
            if constrain_weights:
                centers_track_positions = (adhdp.actor.centers[:,0] + 1) / 2 * len(track.segments)
                adhdp.actor_weight_mask = np.append(np.exp(-np.square(centers_track_positions - win_condition)), 0)
            #print("New win condition:", win_condition)
        
    rl.close_window()
    #plt.plot(learning_c, label="actor")
    #plt.plot(learning_a, label="critic")
    #plt.legend()
    #plt.ylim((0, 3))
