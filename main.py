import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from networks import Network, FFNN, RBFNN, Layer, check_io_gradients
from adhdp import ADHDP, ADHDPState
from track import Track, TrackSegmentArc
from common import *

from typing import Callable, Self
from math import exp, floor

MAX_VEL = 1e20
VEL_SCALE = 100

FORCE = 100
DELTA_TIME = 0.05

def transform2d(inp: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = np.zeros(inp.shape)
    out[:,0] = R[:,0,0]*inp[:,0] + R[:,0,1]*inp[:,1]
    out[:,1] = R[:,1,0]*inp[:,0] + R[:,1,1]*inp[:,1]
    return out

class TrackStateDirect(ADHDPState):
    def __init__(self, p_track: Track, p_internal_state: np.ndarray):
        super().__init__(p_internal_state)
        self.track = p_track
        self.collision_mask = self.internal_state[:,0] != self.internal_state[:,0]
        self.win_mask = self.internal_state[:,0] != self.internal_state[:,0]
    
    def get_initial_control(self) -> np.ndarray:
        tangent = self.track.get_path_dir(self.internal_state)
        return tangent

    def step_forward(self, u: np.ndarray, dt: float) -> Self:
        #u = np.random.normal(0, 0.1, u.shape)
        next_states = np.zeros(self.internal_state.shape)
        next_states[:,:2] = self.internal_state[:,:2] + u*FORCE * dt
        next_states[:,2:4] = u*FORCE
        next_states[:,4] = self.internal_state[:,4]

        # Update track distance
        along_tracks, across_tracks = self.track.get_track_coordinates(next_states, False).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length
            
        collisions = self.track.check_collisions(self.internal_state[:,:5], u, next_states[:,:2] - self.internal_state[:,:2])
        collision_mask = np.logical_not(np.isinf(collisions[:,0]))
        win_mask = self.internal_state[:,4] > 1.5
        next_states[collision_mask,:2] = collisions[collision_mask,:2]
        next_states[collision_mask,2:4] = 0

        # reset next to start (so masks can be used in reward function w/o delay)
        next_states[self.collision_mask] = gen_state(self.track, np.count_nonzero(self.collision_mask), True)
        next_states[self.win_mask] = gen_state(self.track, np.count_nonzero(self.win_mask), True)

        # limits
        next_states[:,2:4] = np.maximum(np.minimum(next_states[:,2:4], MAX_VEL), -MAX_VEL)

        res = TrackStateDirect(self.track, next_states)
        res.collision_mask = collision_mask
        res.win_mask = win_mask
        return res

    def get_s(self) -> np.ndarray:
        s = np.zeros((len(self), 2))
        tc = self.track.get_track_coordinates(self.internal_state, True)
        track_length = sum([x.length for x in self.track.segments])
        s[:,0] = tc[:,0] / track_length * 2.0 - 1.0
        s[:,1] = tc[:,1] * 2 / self.track.track_width
        #s[:,1] *= self.track.track_width /  track_length
        return s

    def get_dsdu(self, dt: float) -> np.ndarray:
        dcartdu = np.eye(2) * FORCE * dt

        dcartds = np.zeros((len(self), 2, 2))
        segment_indices = (np.floor(self.internal_state[:,4]) % len(self.track.segments)).astype(np.int32)
        track_length = sum([x.length for x in self.track.segments]) 
        for i, seg in enumerate(self.track.segments):
            filter = segment_indices == i
            s = seg.get_track_coordinates(self.internal_state[filter,:2])
            tangents = seg.get_tangent_at(self.internal_state[filter,:2])
            seg_length_mid = seg.length
            track_width_half = self.track.track_width/2
            if isinstance(seg, TrackSegmentArc):
                delta_a = seg._a2 - seg._a1
                seg_lengths = seg_length_mid + s[:,1] * delta_a
                seg_lengths = seg_lengths[:,np.newaxis]
            else:
                seg_lengths = seg_length_mid

            fract = seg_length_mid / track_length
            dcartds[filter,0] = tangents * seg_lengths / fract / 2
            dcartds[filter,1] = tangents[:,[1,0]] * np.array([[1,-1]]) * track_width_half

        dsdu = np.zeros((len(self), 2, 2))
        for i in range(len(self)):
            dsdu[i,:2] = np.linalg.solve(dcartds[i], dcartdu)
        
        dsdu *= (1 - self.collision_mask.astype(float)[:,np.newaxis,np.newaxis])
        #dsdu[:,1] *= self.track.track_width / track_length
        return dsdu

    def get_reward(self, adhdp: ADHDP) -> np.ndarray:
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width
        velocity_norm = np.linalg.norm(self.internal_state[:,2:4], axis=1) / VEL_SCALE
        progress = 1 - (self.internal_state[:,4] - 1) / len(self.track.segments)
        return 0.01 - self.win_mask.astype(float)
        return (progress
            ) * (1 - adhdp.gamma) - self.win_mask.astype(float)# + self.collision_mask.astype(float)
    
    def get_positions(self) -> np.ndarray:
        return self.internal_state[:,:5]


class TrackState(ADHDPState):
    def __init__(self, p_track: Track, p_internal_state: np.ndarray):
        super().__init__(p_internal_state)
        self.track = p_track
        self.collision_mask = self.internal_state[:,0] == self.internal_state[:,0]
    
    def get_initial_control(self) -> np.ndarray:
        tangent = self.track.get_path_dir(self.internal_state)
        along, across = self.track.get_track_coordinates(self.internal_state, True).T
        radial = tangent[:,[1,0]] * np.array([[1,-1]])
        wall_avoidance = radial * across[:,np.newaxis] * -0.1 / self.track.track_width
        vel_part = self.internal_state[:,2:4] * 0.01
        return wall_avoidance - vel_part

    def step_forward(self, u: np.ndarray, dt: float) -> Self:
        #u = np.random.normal(0, 0.1, u.shape)
        next_states = np.zeros(self.internal_state.shape)
        next_states[:,:2] = self.internal_state[:,:2] + .5 * u*FORCE * dt*dt + self.internal_state[:,2:4] * dt
        next_states[:,2:4] = self.internal_state[:,2:4] + u*FORCE * dt
        next_states[:,4] = self.internal_state[:,4]

        # Update track distance
        along_tracks, across_tracks = self.track.get_track_coordinates(next_states, False).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length
            
        collisions = self.track.check_collisions(self.internal_state[:,:5], u, next_states[:,:2] - self.internal_state[:,:2])
        collisions_mask = np.logical_not(np.isinf(collisions[:,0]))
        #next_states[collisions_mask,:2] = collisions[collisions_mask,:2] + collisions[collisions_mask,2:] * 1e-3
        #next_states[collisions_mask,2:4] = 0
        next_states[collisions_mask] = gen_state(self.track, np.count_nonzero(collisions_mask), True)

        next_states[:,2:4] = np.maximum(np.minimum(next_states[:,2:4], MAX_VEL), -MAX_VEL)

        #next_states[np.isnan(next_states)] = self.internal_state[np.isnan(next_states)]
        res = TrackState(self.track, next_states)
        res.collision_mask = collisions_mask
        return res

    def get_s(self) -> np.ndarray:
        s = np.zeros((len(self), 4))
        tc = self.track.get_track_coordinates(self.internal_state, True)
        track_length = sum([x.length for x in self.track.segments])
        s[:,0] = np.pi * tc[:,0] / track_length * 2.0 - 1.0
        s[:,1] = tc[:,1] * 2 / self.track.track_width
        s[:,2:4] = self.internal_state[:,2:4] / VEL_SCALE
        return s

    def get_dsdu(self, dt: float) -> np.ndarray:
        dposdu = np.eye(2) * FORCE * dt

        tangents = self.track.get_path_dir(self.internal_state)
        dsdpos = np.zeros((len(self), 2, 2))
        segment_indices = (np.floor(self.internal_state[:,4]) % len(self.track.segments)).astype(np.int32)
        for i in range(len(self.track.segments)):
            track_length = self.track.segments[i].length
            dsdpos[segment_indices==i,0] = tangents[segment_indices==i] / track_length
            dsdpos[segment_indices==i,1] = tangents[segment_indices==i][:,[1,0]] * np.array([[1,-1]]) / self.track.track_width*2

        dsdu = np.zeros((len(self), 4, 2))
        for i in range(len(self)):
            dsdu[i,:2] = dsdpos[i,:2] @ dposdu
            dsdu[i,2:] = np.eye(2) / VEL_SCALE * dt * FORCE
        
        dsdu *= (1 - self.collision_mask.astype(float)[:,np.newaxis,np.newaxis])
        return dsdu

    def get_reward(self, adhdp: ADHDP) -> np.ndarray:
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width
        velocity_norm = np.linalg.norm(self.internal_state[:,2:4], axis=1) / VEL_SCALE
        progress = 1 - (self.internal_state[:,4] / len(self.track.segments))
        return (progress + center_distance
            ) * (1 - adhdp.gamma) + self.collision_mask
    
    def get_positions(self) -> np.ndarray:
        return self.internal_state[:,:5]


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


def gen_state(track: Track, count: int, concentrated: bool) -> np.ndarray:
    state_init = np.zeros((count, 5))
    spawns = np.random.choice(len(track.segments), count) + np.random.uniform(0, 1, count)
    if concentrated:
        spawns = spawns*0 + 1

    scatter_radius = track.track_width / np.sqrt(8)
    state_init[:,:2] = track.evaluate_path(spawns)
    state_init[:,1] += np.random.uniform(-scatter_radius, scatter_radius, state_init[:,1].shape)
    state_init[:,2:4] = np.random.uniform(-1, 1, (count, 2)) * 10
    state_init[:,4] = spawns
    return state_init


def training_loop(track: Track, adhdp: ADHDP) -> None:
    states = TrackState(track, gen_state(track, 20, False))
    for _ in range(300):
        states = adhdp.step_and_learn(states, 0.1)


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

    # Generate states

    states = TrackStateDirect(track, gen_state(track, population, True))
    time = 0

    fig, ax = plt.subplots()

    learning = []

    outer_loop = True
    while outer_loop:
        states = TrackStateDirect(track, gen_state(track, population, True))
        time = 0
        while True:
            dt = DELTA_TIME
            #dt = rl.get_frame_time()
            
            # run adhdp
            if not paused or rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                
                #if adhdp.gamma < 0.99:
                #    adhdp.gamma = 1 - (1-adhdp.gamma) * 0.79

                time += dt

                states: TrackStateDirect = adhdp.step_and_learn(states, dt)
            
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
                adhdp.actor.save_weights_to("norays_vel/actor_rbf.dat")
                adhdp.critic.save_weights_to("norays_vel/critic.dat")
                #adhdp.plant.save_weights_to("norays/plant.dat")
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
    
    actor = RBFNN.grid_spaced(2, 
        np.linspace(-1, 1, 20), 
        #np.linspace(-0.032, 0.032, 5))
        np.linspace(-1, 1, 5))

    #actor = FFNN([
    #    Layer.linear(2),
    #    Layer.tanh(20),
    #    Layer.tanh(2),
    #], (-1, 1), (-1, 1))

    #critic = RBFNN.grid_spaced(1, 
    #    np.linspace(-1, 1, 5), 
    #    np.linspace(-1, 1, 4), 
    #    np.linspace(-1, 1, 3), 
    #    np.linspace(-1, 1, 3))

    critic = FFNN([
        Layer.linear(4),
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(1),
    ], (-1, 1), (-1, 1))

    plant = FFNN([
        Layer.linear(4),
        Layer.tanh(20),
        Layer.tanh(20),
        Layer.linear(2),
    ])

    actor.load_weights_from("norays_vel/actor_prepped_rbf.dat")
    critic.load_weights_from("norays_vel/critic.dat")
    #plant.load_weights_from("norays/plant.dat")

    return actor, critic, plant


def main():
    import matplotlib.pyplot as plt

    population = 30
    track = Track("editing/track1.txt")
    
    adhdp = ADHDP(*generate_networks(), population)
    adhdp.u_offsets = np.random.normal(0, 0.1, (population, 2))
    adhdp.gamma = 0.99

    adhdp.train_actor = True
    adhdp.train_critic = True
    adhdp.train_plant = False
    adhdp.train_actor_on_initial = False

    adhdp.use_plant = False
    adhdp.use_actor = True 
    adhdp.actor_learning_rate = 1e-3
    adhdp.critic_learning_rate = 1e-2
    adhdp.plant_learning_rate = 1e-3
    #check_io_gradients(adhdp.critic)

    #test_states = TrackStateDirect(track, gen_state(track, population, False))
    #test_states.check_dsdu()

    visualize(track, adhdp, population)
    adhdp.plot_critic_gradient(0,1, 0,1)
    adhdp.plot_actor_critic(0,1)
    #adhdp.plot_actor_critic(2,3)
    plt.show()


if __name__ == "__main__":
    main()