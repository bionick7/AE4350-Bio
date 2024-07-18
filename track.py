import numpy as np
import pyray as rl

from common import *
from math import fmod, floor


class TrackWall:
    def __init__(self, **kwargs) -> None:
        self.portal = kwargs.get('portal', -1)
        pass

    def draw(self) -> None:
        pass

    def check_collision_rays(self, ray_origin: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        return np.ones(len(ray_directions)) * np.inf


class TrackWallLine(TrackWall):
    def __init__(self, p_start: np.ndarray, p_end: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.start = p_start
        self.end = p_end

    def draw(self) -> None:
        rl.draw_line(int(self.start[0]), int(self.start[1]),
                     int(self.end[0]), int(self.end[1]), FG)

    def check_collision_rays(self, ray_origin: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        # https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/

        v1 = ray_origin - self.start
        v2 = self.end - self.start
        v3 = np.flip(ray_directions, axis=1) * np.array([-1, 1])
        t1 = np.reciprocal(v3.dot(v2)) * np.cross(v2, v1)
        t2 = np.reciprocal(v3.dot(v2)) * v3.dot(v1)

        t1[t2 < 0] = np.inf
        t1[t2 > 1] = np.inf
        t1[t1 < 0] = np.inf
        return t1
    
    def __repr__(self) -> str:
        return f"[Linear wall from ({self.start[0]}, {self.start[1]}) -> ({self.end[0]}, {self.end[1]})]"


class TrackWallArc(TrackWall):
    def __init__(self, p_center: np.ndarray, p_a1: float, p_a2: float,
                 p_radius: float, **kwargs) -> None:
        super().__init__(**kwargs)

        self.center = p_center
        self.a1 = p_a1
        self.a2 = p_a2
        self.radius = p_radius

        # Define point array
        total_angle = (self.a2 - self.a1) % (2*np.pi)
        if total_angle > np.pi:
            total_angle = np.pi - total_angle
        pts_count = int(abs(total_angle)*10)
        angles = np.linspace(self.a1, self.a1 + total_angle, pts_count)
        pts_x = self.center[0] + np.cos(angles) * self.radius
        pts_y = self.center[1] + np.sin(angles) * self.radius
        self.pts = []
        for i in range(pts_count):
            self.pts.append(rl.Vector2(pts_x[i], pts_y[i]))

    @staticmethod
    def from_endpoints(start: np.ndarray, end: np.ndarray, r: float) -> tuple[np.ndarray, float, float, float]:
        # c_x + r*cos(a1) = p1_x
        # c_y + r*sin(a1) = p1_y
        # c_x + r*cos(a2) = p2_x
        # c_y + r*sin(a2) = p2_y

        dd = end - start
        d = np.linalg.norm(dd)
        radius = abs(r)
        clock_direction = r / radius
        assert d/2 <= radius

        # Normal to d
        n = np.zeros(2)
        n[0] =  dd[1]/d
        n[1] = -dd[0]/d

        n *= -clock_direction

        center = start + dd/2 + n * np.sqrt(max(radius*radius - d*d/4, 0))

        # Figure out angles
        cs_a1 = start - center
        cs_a2 = end - center
        a1 = np.arctan2(cs_a1[1], cs_a1[0])
        a2 = np.arctan2(cs_a2[1], cs_a2[0])

        # Make sure angles are in order

        angular_difference = (a2 - a1) % (2*np.pi)
        if angular_difference > np.pi:
            angular_difference = 2*np.pi - angular_difference
        a2 = a1 + angular_difference * clock_direction

        return center, a1, a2, radius

    def draw(self) -> None:
        rl.draw_line_strip(self.pts, len(self.pts), FG)

    def check_collision_rays(self, ray_origin: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        o =  ray_origin - self.center

        # Quadratic equation
        b = 2*ray_directions.dot(o)        # array
        c = o@o - self.radius*self.radius  # float

        delta = b*b-4*c
        
        t12 = np.ones((2, len(ray_directions)))*np.inf
        t12[0, delta > 0] = 0.5 * (-b[delta > 0] - np.sqrt(delta[delta > 0]))
        t12[1, delta > 0] = 0.5 * (-b[delta > 0] + np.sqrt(delta[delta > 0]))
        t12[t12 < 0] = np.inf

        # Check for angles
        impact1 = ray_origin + (ray_directions.T * t12[0].T).T - self.center
        impact2 = ray_origin + (ray_directions.T * t12[1].T).T - self.center

        angles = t12*0
        angles[0, delta > 0] = np.arctan2(impact1[delta > 0, 1], impact1[delta > 0, 0])
        angles[1, delta > 0] = np.arctan2(impact2[delta > 0, 1], impact2[delta > 0, 0])

        a_l, a_r = self.a1, self.a2
        if a_r < a_l:
            a_l, a_r = a_r, a_l

        angular_diff = (angles - a_l) % (2*np.pi)
        ang_delta = a_r - a_l
        t12[angular_diff > ang_delta] = np.inf

        return np.minimum(t12[0], t12[1])

    def __repr__(self) -> str:
        return f"[circular wall from at ({self.center[0]}, {self.center[1]}), R = {self.radius} a in ({self.a1}, {self.a2})]"


class TrackSegment:
    def __init__(self, p_walls) -> None:
        self.walls = p_walls
        self.length = 0

    def draw(self) -> None:
        for wall in self.walls:
            wall.draw()
    
    def check_collision_rays(self, ray_origin: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
        collisions_search = np.zeros([len(self.walls), len(ray_directions)])
        for wall_index, wall in enumerate(self.walls):
            collisions_search[wall_index] = wall.check_collision_rays(ray_origin, ray_directions)
        collisions = np.min(collisions_search, axis=0)
        return collisions
    
    def get_closest_point_on_track(self, pt: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented in base class")
    
    def get_track_coordinates(self, pt: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented in base class")
    
    def get_tangent_at(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented in base class")

    def evaluate_points(self, evals: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented in base class")


class TrackSegmentLine(TrackSegment):
    def __init__(self, p_start: np.ndarray, p_end: np.ndarray, track_width: float) -> None:
        self.start = p_start
        self.end = p_end

        n = np.flip(p_end - p_start) * np.array([-1, 1])
        n = n / np.linalg.norm(n)
        super().__init__([
            TrackWallLine(p_start + n * track_width/2, p_end + n * track_width/2),
            TrackWallLine(p_start - n * track_width/2, p_end - n * track_width/2),
        ])
        self.length = np.linalg.norm(p_end - p_start)
        self._normalized_dir = (p_end - p_start) / self.length
    
    def get_closest_point_on_track(self, pt: np.ndarray) -> np.ndarray:
        t = (pt - self.start).dot(self._normalized_dir)
        return self.start + self._normalized_dir * t
    
    def get_track_coordinates(self, pt: np.ndarray) -> np.ndarray:
        along = (pt - self.start).dot(self._normalized_dir)
        closest = self.start + self._normalized_dir * along[:,np.newaxis]
        across = np.linalg.norm(pt - closest, axis=1)
        return np.concatenate([along, across]).reshape(2, -1).T
    
    def get_tangent_at(self, pts: np.ndarray) -> np.ndarray:
        res = pts*0
        res[:] = self._normalized_dir
        return res
    
    def evaluate_points(self, evals: np.ndarray) -> np.ndarray:
        return self.start + self.length * evals[:,np.newaxis] * self._normalized_dir


class TrackSegmentArc(TrackSegment):
    def __init__(self, start: np.ndarray, end: np.ndarray, p_radius: float, track_width: float) -> None:
        self._center, self._a1, self._a2, self._radius = TrackWallArc.from_endpoints(start, end, p_radius)
        super().__init__([
            TrackWallArc(self._center, self._a1, self._a2, self._radius + track_width/2),
            TrackWallArc(self._center, self._a1, self._a2, self._radius - track_width/2),
        ])
        self.length = self._radius * (self._a2 - self._a1)
    
    def get_closest_point_on_track(self, pt: np.ndarray) -> np.ndarray:
        rel_pt = pt - self._center
        angle = np.arctan2(rel_pt[1], rel_pt[0])
        #angle = min(max(angle, 0), self._a2 - self._a1)
        return self._center + np.array([np.cos(angle), np.sin(angle)]) * self._radius
    
    def get_track_coordinates(self, pt: np.ndarray) -> np.ndarray:
        rel_pt = pt - self._center
        angle = np.arctan2(rel_pt[:,1], rel_pt[:,0])
        #angle = min(max(angle, 0), self._a2 - self._a1)
        t_ang = (angle - self._a1) % (2*np.pi)
        # Half the outside range is negative
        t_ang[t_ang > (self._a2 - self._a1) / 2 + np.pi] -= np.pi*2
        along = t_ang * self._radius
        cos_sin = np.concatenate((np.cos(angle), np.sin(angle))).reshape(2, -1).T
        closest = self._center + cos_sin * self._radius
        across = np.linalg.norm(pt - closest, axis=1)
        
        return np.concatenate((along, across)).reshape(2, -1).T
    
    def get_tangent_at(self, pts: np.ndarray) -> np.ndarray:
        rel_pts = pts - self._center
        res = pts*0
        mult = 1 / np.linalg.norm(rel_pts, axis=1)
        if self._a1 > self._a2: mult *= -1
        res[:,0] = -rel_pts[:,1] * mult
        res[:,1] = rel_pts[:,0] * mult
        return res
    
    def evaluate_points(self, evals: np.ndarray) -> np.ndarray:
        angles = self._a1 * (1 - evals) + self._a2 * evals
        res = np.zeros((len(evals), 2))
        res[:,0] = self._center[0] + np.cos(angles) * self._radius
        res[:,1] = self._center[1] + np.sin(angles) * self._radius
        return res


class Track:
    def __init__(self, load_path: str|None=None) -> None:
        self.track_width = 90
        self.segments = []
        self.starting_point = np.zeros(2)
        if load_path == None:
            # Simple circle
            pt1, pt2 = np.array([0, -100]), np.array([0, 100])
            self.segments = [
                TrackSegmentArc(pt1, pt2, 100, self.track_width),
                TrackSegmentArc(pt2, pt1, 100, self.track_width)
            ]
            self.starting_point = pt1
        else:
            with open(load_path, 'rt') as f:
                lines = f.readlines()
            active_point = np.zeros(2)
            for line in lines:
                op = line[0]
                args_str = filter(None, line[:-1].split(' ')[1:])
                args = np.array([int(x.strip()) for x in args_str])
                if op == '.':
                    active_point = args
                    self.starting_point = args
                elif op == '|':
                    pt = args[:2]
                    self.segments.append(TrackSegmentLine(active_point, pt, self.track_width))
                    active_point = pt
                elif op == 'c':
                    pt = args[:2]
                    r = args[2]
                    self.segments.append(TrackSegmentArc(active_point, pt, r, self.track_width))
                    active_point = pt

        self.adjacencies = []
        N = len(self.segments)
        for i in range(N):
            self.adjacencies.append([(i+1) % N, (i-1) % N])

    def get_input(self, states: np.ndarray, N_rays: int, R: np.ndarray=np.eye(2)) -> np.ndarray:
        ''' Returns sensor readings evaluated at a certain point 
            states: Nx4 array of generation states
            returns: Nx? array of sensor positions corresponding to positions'''
        N = len(states)
        inp = np.zeros((N, N_rays))
        angles = np.linspace(0, np.pi*2, N_rays+1)[:-1]
        rays = np.zeros((N_rays, 2))

        rays[:,0] = R[0,0]*np.cos(angles) + R[0,1]*np.sin(angles)
        rays[:,1] = R[1,0]*np.cos(angles) + R[1,1]*np.sin(angles)

        for state_index, state in enumerate(states):
            segment_index = int(floor(state[4] % len(self.segments)))
            inp[state_index] = self.check_collision_rays(segment_index, state[:2], rays)
        return inp
    
    def simulation_step(self, states: np.ndarray, outp: np.ndarray, dt: float) -> np.ndarray:
        ''' Advances simulation by one step
            states: Nx4 array of generation states
            outp: Nx2 array with accelerations
            returns: Nx? array of sensor positions corresponding to positions'''
        step = states[:,2:4] * dt + outp * dt*dt*.5
        states[:,2:4] += outp * dt

        step_norms = np.linalg.norm(step, axis=1) + 1e-9  # avoid 0-division
        collision_dirs = (step.T / step_norms.T).T

        for state_index, state in enumerate(states):
            segment_index = int(floor(state[4] % len(self.segments)))
            lap_index = int(floor(state[4]) // len(self.segments))
            collision_dist = self.check_collision_rays(segment_index, state[:2], collision_dirs[np.newaxis, state_index])[0]
            if step_norms[state_index] < collision_dist:
                # No collision
                states[state_index,:2] += step[state_index]
            else:  
                # Collision
                impact_point = state[:2] + collision_dirs[state_index] * collision_dist
                closest = self.segments[segment_index].get_closest_point_on_track(impact_point)
                normal = (closest - impact_point) / np.linalg.norm(closest - impact_point)
                states[state_index,:2] = impact_point + normal * 0.01
                states[state_index,2:4] = normal * 0.1
                
            # Update track distance
            along_track, across_track = self.segments[segment_index].get_track_coordinates(states[np.newaxis, state_index,:2])[0]
            states[state_index,4] = lap_index * len(self.segments) + segment_index +\
                                    along_track / self.segments[segment_index].length

        return states

    def check_collision_rays(self, root_index: int, 
                             origin: np.ndarray, rays: np.ndarray) -> np.ndarray:
        collisions = self.segments[root_index].check_collision_rays(origin, rays)

        checked = [root_index]
        search_queue = self.adjacencies[root_index][:]
        next_search_queue = []
        filter = np.isinf(collisions)
        while len(next_search_queue) > 0 or len(search_queue) > 0:
            while len(search_queue) > 0:
                index = search_queue.pop(0)
                if index not in checked:
                    checked.append(index)
                    next_search_queue += self.adjacencies[index]
                    collision_checks = self.segments[index].check_collision_rays(origin, rays[filter])
                    collisions[filter] = np.minimum(collisions[filter], collision_checks)

            # Each step down the tree: update filter
            search_queue = next_search_queue[:]
            next_search_queue.clear()
            filter = np.isinf(collisions)
            if not np.any(filter):
                return collisions

        return collisions

    def get_path_dir(self, state: np.ndarray) -> np.ndarray:
        segment_indices = (np.floor(state[:,4] % len(self.segments))).astype(np.int32)
        res = np.zeros((len(state), 2))
        for i in range(len(self.segments)):
            res[segment_indices==i] = self.segments[i].get_tangent_at(state[segment_indices==i,:2])
        return res


    def get_track_coordinates(self, state: np.ndarray) -> np.ndarray:
        segment_indices = (np.floor(state[:,4] % len(self.segments))).astype(np.int32)
        res = np.zeros((len(state), 2))
        for i in range(len(self.segments)):
            res[segment_indices==i] = self.segments[i].get_track_coordinates(state[segment_indices==i,:2])
        return res

    def show(self, state: np.ndarray, c=HIGHLIGHT):
        for seg in self.segments:
            seg.draw()

        path_dirs = self.get_path_dir(state)

        for i, pos in enumerate(state[:,:2]):
            if isinstance(c, COLOR_TYPE):
                color = c
            elif isinstance(c, np.ndarray):
                c = (c*255.99).astype(np.uint8)
                color = rl.Color(c[i,0], c[i,1], c[i,2], 255)
            rl.draw_circle(int(pos[0]), int(pos[1]), 4.0, color)
            
            pos2 = pos + state[i,2:4]
            rl.draw_line(int(pos[0]), int(pos[1]), int(pos2[0]), int(pos2[1]), color)

    def evaluate_path(self, spawns: np.ndarray) -> np.ndarray:
        positions = np.zeros((len(spawns), 2))
        segment_indecies = (np.floor(spawns) % len(self.segments)).astype(np.int32)
        for i, segment in enumerate(self.segments):
            positions[segment_indecies==i] = segment.evaluate_points(spawns[segment_indecies==i]-i)
        return positions

    def show_player_rays(self, state: np.ndarray, N_rays: int):
        '''
            Player refers to the first state, which can be controlled manually
            for debugging purposes
        '''
        segment_index = int(floor(state[0,4])) % len(self.segments)
        p_test = state[0,:2]
        angles = np.linspace(0, np.pi*2, N_rays+1)[:-1]
        rays = np.zeros((N_rays, 2))
        rays[:,0] = np.cos(angles)
        rays[:,1] = np.sin(angles)

        ll = self.check_collision_rays(segment_index, p_test, rays)
        #print(ll)
        #print(self.adjacencies)
        for i in range(N_rays):
            dist = min(ll[i], 1e4)  # Make it finite

            impact = p_test + dist * rays[i]
            rl.draw_line_v(
                rl.Vector2(p_test[0], p_test[1]),
                rl.Vector2(impact[0], impact[1]), HIGHLIGHT
            )


if __name__ == "__main__":
    t = Track("editing/track1.txt")
    print(t.segments[0].walls)
    print(t.segments[1].walls)

    origin = np.array([-199, -100])
    angles = np.linspace(0, np.pi*2, 16+1)[:-1]
    rays = np.zeros((16, 2))
    rays[:,0] = np.cos(angles)
    rays[:,1] = np.sin(angles)

    #t.segments[0].check_collision_rays(origin, rays)
    res = t.check_collision_rays(0, origin, rays)
    print(res)
