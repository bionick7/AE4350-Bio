import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from networks import Network, FFNN, RBFNN, Layer
from adhdp import ADHDP, ADHDPState
from track import Track
from common import *

from typing import Callable, Self
from math import exp, floor

POS_MULTIPLIER = 200

VISCOSITY = 0

def transform2d(inp: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = np.zeros(inp.shape)
    out[:,0] = R[:,0,0]*inp[:,0] + R[:,0,1]*inp[:,1]
    out[:,1] = R[:,1,0]*inp[:,0] + R[:,1,1]*inp[:,1]
    return out

class TrackStateRetaining:
    def __init__(self, track: Track, internal_state: np.ndarray, p_spawns: np.ndarray) -> None:
        self.spawns = p_spawns
        self.aim_pts = track.evaluate_path(internal_state[:,4] + 1.0)
        rotations = np.linspace(0, 2*np.pi, len(self.spawns)+1)[:-1] * 0
        self.transf = np.zeros((len(self.spawns), 2, 2))
        self.transf[:,0,0] =  np.cos(rotations)
        self.transf[:,0,1] =  np.sin(rotations)
        self.transf[:,1,0] = -np.sin(rotations)
        self.transf[:,1,1] =  np.cos(rotations)
        # TODO: maybe also flip

class TrackState(ADHDPState):
    def __init__(self, p_track: Track, p_n_rays: int, p_internal_state: np.ndarray, 
                 retaining: TrackStateRetaining):
        super().__init__(p_internal_state)
        self.track = p_track
        self.n_rays = p_n_rays
        self.retaining = retaining
        self.collision_mask = self.internal_state[:,0] == self.internal_state[:,0]
    
    @classmethod
    def _dynamics(cls, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        ''' Also integrates dxdu '''
        force_multiplier = 1

        x_, y, x_dot, y_dot = x[:4]

        acceleration = force_multiplier * u * POS_MULTIPLIER - VISCOSITY * x[2:4]

        prev_dxdt_du = x[5:].reshape(4, 2)
        dxdt_du = np.zeros((4, 2))
        dxdt_du[0:2] = prev_dxdt_du[2:4]
        dxdt_du[2:4] = force_multiplier * np.eye(2)
        dxdt = np.array([x_dot, y_dot, acceleration[0], acceleration[1], 0])

        return np.concatenate((dxdt, dxdt_du.flatten()))

    def step_forward(self, u: np.ndarray, dt: float) -> Self:
        u = transform2d(u, np.transpose(self.retaining.transf, (0, 2, 1)))
        next_states = np.zeros(self.internal_state.shape)
        for i in range(len(self.internal_state)):
            next_states[i] = rk4(self._dynamics, self.internal_state[i], u[i], dt)

        along_tracks, across_tracks = self.track.get_track_coordinates(next_states).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length
            
        collisions = self.track.check_collisions(self.internal_state[:,:5], u, next_states[:,:2] - self.internal_state[:,:2])
        collisions_mask = np.logical_not(np.isinf(collisions[:,0]))
        next_states[collisions_mask,:2] = collisions[collisions_mask,:2] + collisions[collisions_mask,2:] * 1e-3
        next_states[collisions_mask,2:4] = collisions[collisions_mask,2:] * 0
        #next_states[collisions_mask,:2] = self.retaining.spawns[collisions_mask]
        #next_states[collisions_mask,2:4] = 0

        #next_states[np.isnan(next_states)] = self.internal_state[np.isnan(next_states)]
        res = TrackState(self.track, self.n_rays, next_states, self.retaining)
        res.collision_mask = collisions_mask
        return res

    def get_x(self) -> np.ndarray:
        #return self.track.get_input(self.internal_state[:,:4], self.n_rays)
        x = np.zeros((len(self), 4 + self.n_rays))

        # Relative position
        x[:,:2] = (self.internal_state[:,:2] - self.retaining.aim_pts) / POS_MULTIPLIER
        x_norm = np.linalg.norm(x[:,:2], axis=1)
        x[x_norm > 2,:2] = x[x_norm > 2,:2] / x_norm[x_norm > 2,np.newaxis]
        x[:,:2] = transform2d(x[:,:2], self.retaining.transf)

        # Velocity
        x[:,2:4] = self.internal_state[:,2:4] / POS_MULTIPLIER
        x[:,2:4] = transform2d(x[:,2:4], self.retaining.transf)

        self.rays, self.ray_normals = self.track.get_input(self.internal_state[:,:5], self.n_rays, self.retaining.transf)
        rays_outp = np.exp(-4*np.maximum(self.rays, 0)/self.track.track_width)
        x[:,4:] = rays_outp
        #x[:,4:] = rays_outp[:,self.n_rays//2] - rays_outp[self.n_rays//2,:]
        return x

    def get_dxdu(self) -> np.ndarray:
        dxdu = np.zeros((len(self), 4 + self.n_rays, 2))
        #dxdu[:,:4] = self.internal_state[:,5:].reshape(-1, 4, 2) / POS_MULTIPLIER

        force_multiplier = 1
        dt = 0.1
        dxdu[:,:4] = np.array([[force_multiplier*dt*dt*.5, 0],
                               [0, force_multiplier*dt*dt*.5],
                               [force_multiplier*dt, 0],
                               [0, force_multiplier*dt],
                               ]) / POS_MULTIPLIER
        
        angles = np.linspace(0, np.pi*2, self.n_rays+1)[:-1]
        rx, ry = np.cos(angles)[np.newaxis,:], np.sin(angles)[np.newaxis,:]
        nx, ny = self.ray_normals[:,:,0], self.ray_normals[:,:,1]
        r_dot_n = rx*nx + ry*ny + 1e-10
        dray_dx_ = nx / r_dot_n
        dray_dy = ny / r_dot_n
        dray_du = (dray_dx_[:,:,np.newaxis] * dxdu[:,np.newaxis,0] + dray_dy[:,:,np.newaxis] * dxdu[:,np.newaxis,1]) * POS_MULTIPLIER
        
        for i in range(2):
            dxdu[:, :2,i] = transform2d(dxdu[:, :2,i], self.retaining.transf)
            dxdu[:,2:4,i] = transform2d(dxdu[:,2:4,i], self.retaining.transf)
            dray_du[:,:,i] = transform2d(dray_du[:,:,i], self.retaining.transf)

        x_rays = np.exp(-4*np.maximum(self.rays, 0)/self.track.track_width)
        dx_dray = -4/self.track.track_width * x_rays[:,:,np.newaxis]
        dxdu[:,4:] = dray_du * dx_dray
        #dxdu[:,4:] = dray_du * self.track.track_width/2

        #dxdu[:,:4] = 0
        dxdu *= (1 - self.collision_mask.astype(float)[:,np.newaxis,np.newaxis])
        return dxdu

    def get_reward(self, adhdp: ADHDP) -> np.ndarray:
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state[:,:5]).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width
        tgt_distance = np.linalg.norm(self.internal_state[:,:2] - self.retaining.aim_pts, axis=1) / POS_MULTIPLIER
        #return (center_distance + tgt_distance) * (1 - adhdp.gamma) + self.collision_mask.astype(float)
        return center_distance * (1 - adhdp.gamma) + self.collision_mask.astype(float)
        #return self.collision_mask.astype(float)
        #return along_tracks * 0
    
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


def gen_state(track: Track, count: int, n_rays: int) -> TrackState:
    state_init = np.zeros((count, 4*3 + 1))
    spawns = np.random.choice(len(track.segments), count) + np.random.uniform(0, 1, count)
    state_init[:,:2] = track.evaluate_path(spawns)
    state_init[:,2:4] = np.random.uniform(-1, 1, (count, 2)) * 10
    state_init[:,4] = spawns
    retaining = TrackStateRetaining(track, state_init, state_init[:,:2])
    return TrackState(track, n_rays, state_init, retaining)


def training_loop(track: Track, adhdp: ADHDP, N_rays: int=8) -> None:
    states = gen_state(track, 20, N_rays)
    for _ in range(300):
        states = adhdp.step_and_learn(states, 0.1)


def visualize(track: Track, adhdp: ADHDP, population: int, N_rays: int=8):
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

    states = gen_state(track, population, N_rays)
    time = 0

    adhdp.gamma = 0.0

    fig, ax = plt.subplots()

    learning = []

    outer_loop = True
    while outer_loop:
        states = gen_state(track, population, N_rays)
        adhdp.reinitialize(states)
        time = 0
        while True:
            dt = 0.1
            #dt = rl.get_frame_time()
            
            # run adhdp
            if not paused or rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
                
                #if adhdp.gamma < 0.99:
                #    adhdp.gamma = 1 - (1-adhdp.gamma) * 0.79

                time += dt
                states: TrackState = adhdp.step_and_learn(states, dt)
            
                if np.average(states.get_reward(adhdp)) > 100:
                    break
                #if np.median(states.get_reward(adhdp)) < 1e-2:
                #    break
                #if time > 10:
                #    break

                #print(np.average(adhdp.error[1]))
                learning.append(np.average(adhdp.error, axis=0))


            # Drawing
            rl.begin_drawing()
            rl.clear_background(BG)
            rl.begin_mode_2d(camera)
            update_camera(camera)
            
            xx = states.get_positions()
            track.show_player_rays(xx, N_rays)
            track.show(xx)
            for pos in states.retaining.aim_pts:
                rl.draw_circle(int(pos[0]), int(pos[1]), 4.0, rl.GREEN)

            rl.end_mode_2d()
            rl.draw_fps(5, 5)
            rl.draw_text(str(adhdp.J_prev[0]), 5, 20, 16, FG)
            rl.end_drawing()
            
            # User Input
            if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) or rl.is_gamepad_button_pressed(0, 0):
                paused = not paused
            if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
                adhdp.critic.save_weights_to("saved_networks/track1/critic.dat")
                pass
            if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
                break
                
            if rl.window_should_close():
                outer_loop = False
                break
        
        ax.clear()
        learning_array = np.array(learning)
        ax.plot(learning_array[:,0], label="actor")
        ax.plot(learning_array[:,1], label="critic")
        ax.plot(learning_array[:,2], label="plant")
        ax.set_ylim((0, 3))
        ax.grid()
        ax.legend()
        plt.pause(1e-10)
        
    rl.close_window()
    #plt.plot(learning_c, label="actor")
    #plt.plot(learning_a, label="critic")
    #plt.legend()
    #plt.ylim((0, 3))


def generate_networks(N_rays: int) -> tuple[Network, Network, Network]:
    #actor = FFNN([
    #    Layer.linear(4),
    #    Layer.tanh(4),
    #    Layer.tanh(2),
    #], (-1, 1), (0, 0))
    #actor.load_weights_from("saved_networks/track1/actor_start.dat")

    actor = FFNN([
        Layer.linear(4 + N_rays),
        #Layer.tanh(4 + N_rays),
        Layer.tanh(2),
    ], (-1, 1), (-1, 1))
    #actor.set_weight_matrix(0, np.hstack((np.eye(4 + N_rays), np.zeros((4 + N_rays, 1)))))
    actor.set_weight_matrix(-1, np.array([
        [-0.5, 0, -1, 0] + [-1,0, 0, 1, 0, 0] + [0],
        [0, -0.5, 0, -1] + [0,-1,-1, 0, 1, 1] + [0]
    ]))

    #actor._initialize()
    
    old_critic = FFNN([
        Layer.linear(6),
        Layer.tanh(10), 
        Layer.tanh(10), 
        Layer.linear(1),
    ], (-1, 1), (-1, 1))
    old_critic.load_weights_from("saved_networks/track1/critic_start.dat")

    new_critic_part = FFNN([
        Layer.linear(N_rays),
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(1),
    ], (-1, 1), (-1, 1))
    critic = FFNN.stack(old_critic, new_critic_part)

    W = critic.get_weight_matrix(0)
    W[:,-3:-1] = W[:,4:6]  # copy output
    W[:10,4:-3] = 0
    W[10:21,4:-3] = np.random.uniform(-1, 1, (10, N_rays))
    critic.set_weight_matrix(0,W)
    
    # Combine output
    Wout = critic.get_weight_matrix(2)
    critic.architecture[-1].size = 1
    critic.weights[critic.offsets[-1]:critic.offsets[-1]+21] = np.sum(Wout, axis=0)
    critic.weights = np.resize(critic.weights, critic.offsets[-1]+21)

    plant = FFNN([
        Layer.linear(N_rays + 6),
        Layer.sigmoid(N_rays*2),
        Layer.sigmoid(N_rays*2),
        Layer.linear(N_rays + 4),
    ])

    #critic.load_weights_from("saved_networks/track1/critic.dat")
    #critic.load_weights_from("saved_networks/track1/actor_gamma0.dat")

    return actor, critic, plant


def main():
    import matplotlib.pyplot as plt

    population = 20
    N_rays = 6
    track = Track("editing/track1.txt")
    
    adhdp = ADHDP(*generate_networks(N_rays), population)
    adhdp.gamma = 0.99

    adhdp.train_actor = True
    adhdp.train_critic = True
    adhdp.train_plant = False
    adhdp.use_plant = False
    adhdp.actor_learning_rate = 1e-3
    adhdp.critic_learning_rate = 1e-3
    adhdp.plant_learning_rate = 1e-2

    visualize(track, adhdp, population, N_rays)
    #adhdp.plot_actor_critic(4,7)

    plt.show()

if __name__ == "__main__":
    main()