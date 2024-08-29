from common import *
from track import Track, TrackSegmentArc
from adhdp import ADHDPState, ADHDP

from typing import Callable, Self
from math import exp, floor

def transform2d(inp: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = np.zeros(inp.shape)
    out[:,0] = R[:,0,0]*inp[:,0] + R[:,0,1]*inp[:,1]
    out[:,1] = R[:,1,0]*inp[:,0] + R[:,1,1]*inp[:,1]
    return out


def get_dcartds(track: Track, internal_state: np.ndarray) -> np.ndarray:
    ''' Returns the derivative of track coordinates w.r.t cartesian coordinates '''
    dcartds = np.zeros((len(internal_state), 2, 2))
    segment_indices = (np.floor(internal_state[:,4]) % len(track.segments)).astype(np.int32)
    track_length = sum([x.length for x in track.segments]) 

    for i, seg in enumerate(track.segments):
        segment_filter = segment_indices == i
        s = seg.get_track_coordinates(internal_state[segment_filter,:2])
        tangents = seg.get_tangent_at(internal_state[segment_filter,:2])
        seg_length_mid = seg.length
        track_width_half = track.track_width/2

        if isinstance(seg, TrackSegmentArc):
            # adjust for turning radius
            delta_a = seg._a2 - seg._a1
            seg_lengths = seg_length_mid + s[:,1] * delta_a
            seg_lengths = seg_lengths[:,np.newaxis]
        else:
            seg_lengths = seg_length_mid

        fract = seg_length_mid / track_length
        dcartds[segment_filter,:,0] = tangents * seg_lengths / fract / 2
        dcartds[segment_filter,:,1] = tangents[:,[1,0]] * np.array([[1,-1]]) * track_width_half
        
    return dcartds


def gen_state(track: Track, count: int, concentrated: bool) -> np.ndarray:
    state_init = np.zeros((count, 5))
    spawns = np.random.choice(len(track.segments), count) + np.random.uniform(0, 1, count)
    if concentrated:
        spawns = spawns*0 + 1

    scatter_radius = track.track_width / 2.01
    state_init[:,:2] = track.evaluate_path(spawns)
    state_init[:,:2] += np.random.uniform(-scatter_radius, scatter_radius, state_init[:,:2].shape)
    state_init[:,2:4] = np.random.uniform(-1, 1, (count, 2)) * 0
    state_init[:,4] = spawns
    return state_init


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
        next_states[:,:2] = self.internal_state[:,:2] + u*self.config.get("force", 1) * dt
        next_states[:,2:4] = u*self.config.get("force", 1)
        next_states[:,4] = self.internal_state[:,4]

        # Update track distance
        along_tracks, across_tracks = self.track.get_track_coordinates(next_states, False).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length
            
        # Check collisions
        collisions = self.track.check_collisions(self.internal_state[:,:5], u, next_states[:,:2] - self.internal_state[:,:2])
        collision_mask = np.logical_not(np.isinf(collisions[:,0]))
        win_mask = self.internal_state[:,4] > 3.9
        next_states[collision_mask,:2] = collisions[collision_mask,:2]
        next_states[collision_mask,2:4] = 0

        # reset next to start (so masks can be used in reward function w/o delay)
        next_states[self.collision_mask] = gen_state(self.track, np.count_nonzero(self.collision_mask), True)
        next_states[self.win_mask] = gen_state(self.track, np.count_nonzero(self.win_mask), True)

        # limits
        max_vel = self.config.get("max_vel")
        next_states_vel_norm = np.linalg.norm(next_states[:,2:4], axis=1)
        next_states[next_states_vel_norm > max_vel,2:4] = next_states[next_states_vel_norm > max_vel,2:4] / next_states_vel_norm[next_states_vel_norm > max_vel,np.newaxis] * max_vel

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

    def get_dsdu(self, dt: float, u: np.ndarray) -> np.ndarray:
        dcartdu = np.eye(2) * self.config.get("force", 1) * dt

        dcartds = get_dcartds(self.track, self.internal_state)

        dsdu = np.zeros((len(self), 2, 2))
        for i in range(len(self)):
            dsdu[i] = np.linalg.solve(dcartds[i], dcartdu)
        
        dsdu *= (1 - self.collision_mask.astype(float)[:,np.newaxis,np.newaxis])
        return dsdu

    def get_reward(self) -> np.ndarray:
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width
        velocity_norm = np.linalg.norm(self.internal_state[:,2:4], axis=1) / self.config.get("vel_scale", 1)
        progress = 1 - (self.internal_state[:,4] - 1) / len(self.track.segments)
        #return 0.01 - self.win_mask.astype(float) + self.collision_mask.astype(float)
        return (progress
            ) * (1 - self.config["gamma"]) - self.win_mask.astype(float) + self.collision_mask.astype(float)
    
    def get_positions(self) -> np.ndarray:
        return self.internal_state[:,:5]


class TrackState(ADHDPState):
    def __init__(self, p_track: Track, p_internal_state: np.ndarray):
        super().__init__(p_internal_state)
        self.track = p_track
        self.collision_mask = self.internal_state[:,0] != self.internal_state[:,0]
        self.win_mask = self.internal_state[:,0] != self.internal_state[:,0]
    
    def get_initial_control(self) -> np.ndarray:
        tangent = self.track.get_path_dir(self.internal_state)
        along, across = self.track.get_track_coordinates(self.internal_state, True).T
        radial = tangent[:,[1,0]] * np.array([[1,-1]])
        wall_avoidance = radial * across[:,np.newaxis] * -0.1 / self.track.track_width
        vel_part = self.internal_state[:,2:4] * 0.01
        return wall_avoidance - vel_part + tangent * 0.2

    def step_forward(self, u: np.ndarray, dt: float) -> Self:
        #u = np.random.normal(0, 0.1, u.shape)
        next_states = np.zeros(self.internal_state.shape)
        next_states[:,:2] = self.internal_state[:,:2] + .5 * u*self.config.get("force", 1) * dt*dt + self.internal_state[:,2:4] * dt
        next_states[:,2:4] = self.internal_state[:,2:4] + u*self.config.get("force", 1) * dt
        next_states[:,4] = self.internal_state[:,4]

        # Update track distance
        along_tracks, across_tracks = self.track.get_track_coordinates(next_states, False).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length
            
        # Check collisions
        collisions = self.track.check_collisions(self.internal_state[:,:5], u, next_states[:,:2] - self.internal_state[:,:2])
        collision_mask = np.logical_not(np.isinf(collisions[:,0]))
        win_mask = self.internal_state[:,4] > 2
        next_states[collision_mask,:2] = collisions[collision_mask,:2]
        next_states[collision_mask,2:4] = 0

        # reset next to start (so masks can be used in reward function w/o delay)
        next_states[self.collision_mask] = gen_state(self.track, np.count_nonzero(self.collision_mask), True)
        next_states[self.win_mask] = gen_state(self.track, np.count_nonzero(self.win_mask), True)

        # limits
        max_vel = self.config.get("max_vel")
        next_states_vel_norm = np.linalg.norm(next_states[:,2:4], axis=1)
        next_states[next_states_vel_norm > max_vel,2:4] = next_states[next_states_vel_norm > max_vel,2:4] / next_states_vel_norm[next_states_vel_norm > max_vel,np.newaxis] * max_vel

        res = TrackState(self.track, next_states)
        res.collision_mask = collision_mask
        res.win_mask = win_mask
        return res

    def get_s(self) -> np.ndarray:
        s = np.zeros((len(self), 4))
        #s = np.zeros((len(self), 2))
        tc = self.track.get_track_coordinates(self.internal_state, True)
        track_length = sum([x.length for x in self.track.segments])
        s[:,0] = np.pi * tc[:,0] / track_length * 2.0 - 1.0
        s[:,1] = tc[:,1] * 2 / self.track.track_width
        s[:,2:4] = self.internal_state[:,2:4] / self.config.get("vel_scale", 1)
        return s

    def get_dsdu(self, dt: float, u: np.ndarray) -> np.ndarray:
        dcartdu = np.eye(2) * self.config.get("force", 1) * dt*dt*0.5

        dcartds = get_dcartds(self.track, self.internal_state)
        dcartds[:,:,0] *= 1/np.pi  # Determined empirically

        dsdu = np.zeros((len(self), 4, 2))
        for i in range(len(self)):
            dsdu[i,:2] = np.linalg.solve(dcartds[i], dcartdu)
            dsdu[i,2:] = np.eye(2) * self.config.get("force", 1) / self.config.get("vel_scale", 1)
        
        dsdu *= (1 - self.collision_mask.astype(float)[:,np.newaxis,np.newaxis])
        return dsdu

    def get_reward(self) -> np.ndarray:
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width
        velocity_norm = np.linalg.norm(self.internal_state[:,2:4], axis=1) / self.config.get("vel_scale", 1)
        progress = 1 - (self.internal_state[:,4] / len(self.track.segments) - 1)
        return (progress
            ) * (1 - self.config["gamma"]) - self.win_mask.astype(float) + self.collision_mask.astype(float)
    
    def get_positions(self) -> np.ndarray:
        return self.internal_state[:,:5]


class TrackStateRot(ADHDPState):
    def __init__(self, p_track: Track, p_internal_state: np.ndarray):
        super().__init__(p_internal_state)
        self.track = p_track
        self.collision_impact = np.zeros(len(self))
        self.win_mask = self.internal_state[:,0] != self.internal_state[:,0]
        self.collision_mask = self.internal_state[:,0] != self.internal_state[:,0]
        self.win_condition: float = 4
        self.reset = self.internal_state[:,0] != self.internal_state[:,0]
        self.cartesian_offset = np.zeros((len(self), 2))
    
    def get_initial_control(self) -> np.ndarray:
        tangents = self.track.get_path_dir(self.internal_state)
        rots = np.arctan2(tangents[:,1], tangents[:,0])
        rots = np.mod(rots, np.pi*2) + 0.5
        return rots

    def step_forward(self, u: np.ndarray, dt: float) -> Self:
        #u = np.random.normal(0, 0.1, u.shape)
        next_states = np.zeros(self.internal_state.shape)
        force = np.zeros((len(self), 2))
        force[:,0] = np.cos(u[:,0]) * self.config.get("force", 1)
        force[:,1] = np.sin(u[:,0]) * self.config.get("force", 1)
        force += self.cartesian_offset * self.config.get("force", 1)
        next_states[:,:2] = self.internal_state[:,:2] + force * dt*dt*0.5 + self.internal_state[:,2:4] * dt
        next_states[:,2:4] = self.internal_state[:,2:4] + force * dt
        next_states[:,4] = self.internal_state[:,4]

        # Update track distance
        along_tracks, across_tracks = self.track.get_track_coordinates(next_states, False).T
        for i in range(len(self.internal_state)):
            segment_index = int(floor(self.internal_state[i, 4] % len(self.track.segments)))
            lap_index = int(floor(self.internal_state[i, 4]) // len(self.track.segments))
            next_states[i,4] = lap_index * len(self.track.segments) + segment_index +\
                               along_tracks[i] / self.track.segments[segment_index].length
            
        # Check collisions
        collisions = self.track.check_collisions(self.internal_state[:,:5], u, next_states[:,:2] - self.internal_state[:,:2])
        collision_mask = np.logical_not(np.isinf(collisions[:,0]))
        win_mask = self.internal_state[:,4] > self.win_condition
        collision_impact = np.zeros(len(self))
        collision_impact[collision_mask] = (
            - collisions[collision_mask,2] * self.internal_state[collision_mask,2]
            - collisions[collision_mask,3] * self.internal_state[collision_mask,3])
        next_states[collision_mask,:2] = collisions[collision_mask,:2] + collisions[collision_mask,2:4]
        next_states[collision_mask,2:4] = 0

        # reset next to start (so masks can be used in reward function w/o delay)
        #next_states[self.collision_impact > 0] = gen_state(self.track, np.count_nonzero(self.collision_impact > 0), True)
        #next_states[self.win_mask] = gen_state(self.track, np.count_nonzero(self.win_mask), True)

        # limits
        max_vel = self.config.get("max_vel")
        next_states_vel_norm = np.linalg.norm(next_states[:,2:4], axis=1)
        next_states[next_states_vel_norm > max_vel,2:4] = next_states[next_states_vel_norm > max_vel,2:4] / next_states_vel_norm[next_states_vel_norm > max_vel,np.newaxis] * max_vel
        next_states[self.reset] = self.internal_state[self.reset]

        res = TrackStateRot(self.track, next_states)
        res.reset = self.internal_state[:,0] != self.internal_state[:,0]#np.logical_or(np.logical_or(self.collision_mask, self.win_mask), self.reset)
        res.collision_impact = collision_impact
        res.collision_mask = collision_mask
        res.win_mask = win_mask
        res.win_condition = self.win_condition
        res.cartesian_offset = self.cartesian_offset
        return res

    def get_s(self) -> np.ndarray:
        s = np.zeros((len(self), 2))
        tc = self.track.get_track_coordinates(self.internal_state, True)
        track_length = sum([x.length for x in self.track.segments])
        s[:,0] = tc[:,0] / track_length * 2.0 - 1.0
        s[:,1] = tc[:,1] * 2 / self.track.track_width
        #s[:,1] *= self.track.track_width /  track_length
        return s

    def get_dsdu(self, dt: float, u: np.ndarray) -> np.ndarray:
        dcartdu = np.zeros((len(self), 2))
        dcartdu[:,0] = -np.sin(u[:,0]) * self.config.get("force", 1) * dt*dt*.5
        dcartdu[:,1] =  np.cos(u[:,0]) * self.config.get("force", 1) * dt*dt*.5

        dcartds = get_dcartds(self.track, self.internal_state)

        dsdu = np.zeros((len(self), 2, 1))
        for i in range(len(self)):
            dsdu[i] = np.linalg.solve(dcartds[i], dcartdu[i])[:,np.newaxis]
                
        dsdu *= (1 - self.reset.astype(float)[:,np.newaxis,np.newaxis])
        return dsdu

    def get_reward(self) -> np.ndarray:
        along_tracks, across_tracks = self.track.get_track_coordinates(self.internal_state).T
        center_distance = 2*np.abs(across_tracks) / self.track.track_width
        velocity_norm = np.linalg.norm(self.internal_state[:,2:4], axis=1) / self.config.get("vel_scale", 1)
        progress = 1 - (self.internal_state[:,4] - 1) / len(self.track.segments)
        #return 0.01 - self.win_mask.astype(float) + self.collision_mask.astype(float)
        res = progress * (1 - self.config["gamma"])
        #return res
        return (progress + np.minimum(self.collision_impact / self.config.get("vel_scale", 1), 1)
            ) * (1 - self.config["gamma"])# - self.win_mask.astype(float)
    
    def get_positions(self) -> np.ndarray:
        return self.internal_state[:,:5]
