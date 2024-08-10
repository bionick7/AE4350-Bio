import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.markers

from track import Track
from track_states import TrackStateRot, gen_state
from networks import RBFNN
from main import generate_networks
from common import *

def get_track_outline(track: Track, pts_cout: int=100) -> tuple[np.ndarray, np.ndarray]:
    outer_track = np.zeros((pts_cout, 2))
    outer_track[:,0] = np.linspace(0, track.track_length, pts_cout)
    outer_track[:,1] = track.track_width/2
    outer_cart = track.track_to_cartesian_coords(outer_track)

    inner_track = np.zeros((pts_cout, 2))
    inner_track[:,0] = np.linspace(0, track.track_length, pts_cout)
    inner_track[:,1] =-track.track_width/2
    inner_cart = track.track_to_cartesian_coords(inner_track)
    return inner_cart, outer_cart


def show_track(track: Track, filename: str|None=None):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots(1, 1)
    if not isinstance(ax, Axes): return

    for outline in get_track_outline(track):
        ax.plot(outline[:,0], outline[:,1], 'k-')

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)


def show_actor_rbfs(track: Track, filename: str|None=None):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots(1, 1)
    if not isinstance(ax, Axes): return

    actor_nn, *_ = generate_networks()
    if not isinstance(actor_nn, RBFNN): return

    centers_path_coordinates = actor_nn.centers * np.array([[track.track_length/2, track.track_width/2]])
    centers_path_coordinates[:,0] += track.track_length/2

    cartesian = track.track_to_cartesian_coords(centers_path_coordinates)
    for outline in get_track_outline(track):
        ax.plot(outline[:,0], outline[:,1], 'k-')
    ax.plot(cartesian[:,0], cartesian[:,1], 'rx')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)


def show_actorcritic_track(track: Track, max_vel:float, actor_weights_file: str, 
                           critic_weights_file: str, filename: str|None=None):
    fig = plt.figure(figsize=(6, 5))
    ax_actor, ax_critic = fig.subplots(2, 1, sharex=True)

    # Load networks
    actor_nn, critic_nn, _ = generate_networks()
    actor_nn.load_weights_from(actor_weights_file)
    critic_nn.load_weights_from(critic_weights_file)

    # Get track-coordinates
    #start_pt = -1 + track.segments[0].length / track.track_length *2 - 0.1
    start_pt = -1
    xi, eta = np.meshgrid(
        np.linspace(start_pt, 1, int(track.track_length/5)), 
        np.linspace(-1, 1, int(track.track_width/5))
    )
    field_track_coords = np.array([xi.flatten(), eta.flatten()]).T

    # Evaluate functions
    zz_actor = np.zeros(len(field_track_coords))
    zz_critic = np.zeros(len(field_track_coords))
    for i, coord in enumerate(field_track_coords):
        zz_actor[i] = actor_nn.eval(coord)[0]
        zz_critic[i] = critic_nn.eval(np.append(coord, zz_actor[i]))[0]

    # Initialize trackstates
    population = 4
    
    initial = np.zeros((population, 5))
    initial[:,:2] = track.evaluate_path(np.ones(4))
    initial[:,1] += np.linspace(-.5, .5, population+2)[1:-1] * track.track_width
    initial[:,4] = np.ones(4)
    states = TrackStateRot(track, initial)
    states.config = {"max_vel": max_vel, "force": 100}

    # Evaluate paths
    paths = np.zeros((population, 40, 2))
    path_us = np.zeros((population, 40))
    for i in range(40):
        paths[:,i] = states.get_s()
        for j in range(population):
            path_us[j, i] = actor_nn.eval(paths[j,i])[0]
        states = states.step_forward(path_us[:, i, np.newaxis], 0.5)
        states.config = {"max_vel": max_vel, "force": 100}

    # Transform to cartesian coordinates
    field_path_coordinates = field_track_coords * np.array([[track.track_length/2, track.track_width/2]])
    field_path_coordinates[:,0] += track.track_length/2
    field_cartesian = track.track_to_cartesian_coords(field_path_coordinates)

    paths_path_coordinates = paths * np.array([[[track.track_length/2, track.track_width/2]]])
    paths_path_coordinates[:,:,0] += track.track_length/2
    paths_cartesian = np.zeros(paths.shape)
    for i in range(population):
        paths_cartesian[i] = track.track_to_cartesian_coords(paths_path_coordinates[i])

    # Draw outline
    for outline in get_track_outline(track):
        ax_actor.plot(outline[:,0], outline[:,1], 'k-')
        ax_critic.plot(outline[:,0], outline[:,1], 'k-')

    # Draw contours
    contours_actor = ax_actor.contourf(
        field_cartesian[:,0].reshape(xi.shape), field_cartesian[:,1].reshape(xi.shape), zz_actor.reshape(xi.shape),
        np.arange(0, 8, 0.5)
    )
    contours_critic = ax_critic.contourf(
        field_cartesian[:,0].reshape(xi.shape), field_cartesian[:,1].reshape(xi.shape), zz_critic.reshape(xi.shape)
    )
    
    # Draw paths
    for i in range(population):
        color = ['r', 'g', 'b', 'y'][i]
        ax_actor.plot(paths_cartesian[i,:,0], paths_cartesian[i,:,1], color=color)
        for j in range(40):
            ax_actor.scatter(paths_cartesian[i,j,0], paths_cartesian[i,j,1], 
                             marker=(1, 2, path_us[i,j] * RAD2DEG - 90), color=color, s=100)
    
    # Legend, axes, ...
    fig.colorbar(contours_actor)
    fig.colorbar(contours_critic)
    ax_actor.set_ylabel("y")
    ax_critic.set_xlabel("x")
    ax_critic.set_ylabel("y")
    ax_actor.set_aspect("equal")
    ax_critic.set_aspect("equal")

    fig.tight_layout()

    # Save figure
    if filename is not None:
        fig.savefig(filename)


def show_critic_derivative_u(track: Track, actor_weights_file: str, critic_weights_file: str, filename: str|None=None):
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    if not isinstance(ax, Axes): return
    
    # Load networks
    actor_nn, critic_nn, _ = generate_networks()
    actor_nn.load_weights_from(actor_weights_file)
    critic_nn.load_weights_from(critic_weights_file)
    
    # Get track-coordinates
    #start_pt = -1 + track.segments[0].length / track.track_length *2 - 0.1
    start_pt = -1
    xi, eta = np.meshgrid(
        np.linspace(start_pt, 1, int(track.track_length/5)), 
        np.linspace(-1, 1, int(track.track_width/5))
    )
    field_track_coords = np.array([xi.flatten(), eta.flatten()]).T

    # Evaluate functions
    zz_critic = np.zeros(len(field_track_coords))
    for i, coord in enumerate(field_track_coords):
        u = actor_nn.eval(coord)[0]
        zz_critic[i] = critic_nn.get_io_gradient(np.append(coord, u))[-1,0]
    
    # Transform to cartesian coordinates
    field_path_coordinates = field_track_coords * np.array([[track.track_length/2, track.track_width/2]])
    field_path_coordinates[:,0] += track.track_length/2
    field_cartesian = track.track_to_cartesian_coords(field_path_coordinates)

    
    # Draw outline
    for outline in get_track_outline(track):
        ax.plot(outline[:,0], outline[:,1], 'k-')

    # Draw contours
    contours = ax.contourf(
        field_cartesian[:,0].reshape(xi.shape), field_cartesian[:,1].reshape(xi.shape), zz_critic.reshape(xi.shape),
        np.arange(-1, 1, 0.2)
    )
    
    # Legend, axes, ...
    fig.colorbar(contours)
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_aspect("equal")

    fig.tight_layout()

    # Save figure
    if filename is not None:
        fig.savefig(filename)



def plot_learning_curve(filename: str|None=None):
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    if not isinstance(ax, Axes): return

    learning_array = np.zeros((3, 3, 49))
    for mvel_index, mvel in enumerate([50, 100, np.inf]):
        for run in range(3):
            learning_array[mvel_index, run] = np.loadtxt(f"results/learning_mvel{mvel}_r{run+1}.dat")[:-1]

    for mvel_index, mvel in enumerate([50, 100, np.inf]):
        format = ["r-", "g-", "b-"][mvel_index]
        label = [
            "$v_{max} = 50 m/s$",
            "$v_{max} = 100 m/s$",
            "$v_{max} = \\infty$",
        ][mvel_index]

        best_index = np.argmax(np.average(learning_array[mvel_index,:,25:], axis=1))
        ax.plot(learning_array[mvel_index, best_index] / 4, format, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("progress")
    ax.grid()
    ax.legend()
    #ax.set_ylim(0, 4.6)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)


def print_run_table_latex():
    template = (r"\begin{{tabular}}{{|l|c|c|c|}} \hline" "\n"
                r"                       & Run 1 & Run 2 & Run 3 \\ \hline" "\n"
                r"    $v_{{max}} = 50$     & {0:5.3f} & {1:5.3f} & {2:5.3f} \\ \hline" "\n"
                r"    $v_{{max}} = 100$    & {3:5.3f} & {4:5.3f} & {5:5.3f} \\ \hline" "\n"
                r"    $v_{{max}} = \infty$ & {6:5.3f} & {7:5.3f} & {8:5.3f} \\ \hline" "\n"
                r"\end{{tabular}}" "\n")
    
    learning_array = np.zeros((3, 3, 49))
    for mvel_index, mvel in enumerate([50, 100, np.inf]):
        for run in range(3):
            learning_array[mvel_index, run] = np.loadtxt(f"results/learning_mvel{mvel}_r{run+1}.dat")[:-1]
    learning_performance = np.average(learning_array[:,:,25:], axis=2) / 4
    print(template.format(*learning_performance.flatten()))


def main():
    track = Track("editing/track1.txt")
    #show_track(track, "results/track.png")
    #show_actor_rbfs(track, "results/track_rbf_distribution.png")
    #show_actorcritic_track(track, 50, "final/actor_mvel50_r2.dat", "final/critic_mvel50_r2.dat", "results/mvel50_r2_heigtmaps.png")
    #show_actorcritic_track(track, np.inf, "final/actor_mvelinf_r3.dat", "final/critic_mvelinf_r3.dat", "results/mvelinf_r3_heigtmaps.png")
    show_critic_derivative_u(track, "final/actor_mvelinf_r3.dat", "final/critic_mvelinf_r3.dat")
    show_critic_derivative_u(track, "final/actor_mvel50_r2.dat", "final/critic_mvel50_r2.dat")
    #plot_learning_curve("results/learning_curve.png")
    #print_run_table_latex()


if __name__ == "__main__":
    main()
    plt.show()