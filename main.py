import numpy as np
import matplotlib.pyplot as plt
import pyray as rl

from track_states import TrackStateRot, gen_state
from networks import Network, FFNN, RBFNN, Layer, check_io_gradients
from adhdp import ADHDP, ADHDPState
from track import Track, TrackSegmentArc
from visualization import visualize_adhdp
from common import *

from copy import copy
from math import exp

GAMMA = 0.9


def generate_networks() -> tuple[Network, Network, Network]:
    
    actor = RBFNN.grid_spaced(1,
        np.linspace(-1, 1, 30), 
        np.linspace(-1, 1, 5),
        stdev_mult=2)

    #actor = FFNN([
    #    Layer.linear(2),
    #    Layer.tanh(10),
    #    Layer.tanh(10),
    #    Layer.linear(1),
    #], (-1, 1), (-1, 1))

    #critic = RBFNN.grid_spaced(1, 
    #    np.linspace(-1, 1, 6), 
    #    np.linspace(-1, 1, 4), 
    #    np.linspace(-1, 1, 5))

    critic = FFNN([
        Layer.linear(3),
        Layer.tanh(8),
        Layer.tanh(8),
        Layer.linear(1),
    ], (-1, 1), (-1, 1))

    plant = FFNN([
        Layer.linear(3), 
        Layer.tanh(10),
        Layer.tanh(10),
        Layer.linear(2),
    ])

    #plant.load_weights_from("norays_vel/plant.dat")

    return actor, critic, plant


def main():
    import matplotlib.pyplot as plt

    population = 30
    track = Track("editing/track1.txt")
    
    adhdp = ADHDP(*generate_networks(), population)
    adhdp.actor.load_weights_from("norays_rot/actor_prepped.dat")
    adhdp.critic.load_weights_from("norays_rot/critic.dat")
    adhdp.u_offsets = np.random.normal(0, 0.1, (population, 1))
    adhdp.u_offsets[0] = 0
    adhdp.gamma = GAMMA

    adhdp.train_actor = True
    adhdp.train_critic = True

    adhdp.actor_save_path = "norays_rot/actor.dat"
    adhdp.critic_save_path = "norays_rot/critic.dat"

    adhdp.use_plant = False
    adhdp.actor_learning_rate = 5e-3
    adhdp.critic_learning_rate = 1e-2

    #check_io_gradients(adhdp.critic)
    #test_states = TrackStateRot(track, gen_state(track, population, False))
    #test_states.check_dsdu(1)

    visualize_adhdp(track, adhdp, population, True)
    adhdp.plot_critic_gradient(0,1, 0)
    adhdp.plot_actor_critic(0,1)
    plt.show()


def pretrain():
    import matplotlib.pyplot as plt

    population = 30
    track = Track("editing/track1.txt")
    
    adhdp = ADHDP(*generate_networks(), population)
    adhdp.u_offsets = np.random.normal(0, 0.05, (population, 1))
    adhdp.u_offsets[0] = 0
    adhdp.gamma = GAMMA

    adhdp.train_critic = True
    adhdp.train_actor_on_initial = True

    adhdp.actor_save_path = "norays_rot/actor_prepped_ffnn.dat"
    adhdp.critic_save_path = "norays_rot/critic.dat"
    
    adhdp.use_plant = False
    adhdp.actor_learning_rate = 1e-2
    adhdp.critic_learning_rate = 1e-2

    # Supervised training
    for epoch in range(100):
        EPOCH_SIZE = 100
        states = TrackStateRot(track, gen_state(track, EPOCH_SIZE, False))
        inp = states.get_s()
        out = states.get_initial_control()

        error = 0
        for i in range(EPOCH_SIZE):
            _, E = adhdp.actor.learn_gd(1e-2, inp[i], out[i])
            error += E/EPOCH_SIZE
        print(error)

    visualize(track, adhdp, population, False)
    adhdp.plot_critic_gradient(0,1, 0)
    adhdp.plot_actor_critic(0,1)
    plt.show()


def post_train():
    import matplotlib.pyplot as plt

    population = 30
    track = Track("editing/track1.txt")
    
    adhdp = ADHDP(*generate_networks(), population)
    adhdp.actor.load_weights_from("norays_rot/actor.dat")
    adhdp.critic.load_weights_from("norays_rot/critic_post.dat")
    adhdp.u_offsets = np.random.normal(0, 0.01, (population, 1))
    adhdp.u_offsets[0] = 0
    adhdp.gamma = GAMMA

    adhdp.train_actor = True
    adhdp.train_critic = True

    adhdp.actor_save_path = "norays_rot/actor_post.dat"
    adhdp.critic_save_path = "norays_rot/critic_post.dat"

    adhdp.use_plant = False
    adhdp.actor_learning_rate = 1e-2
    adhdp.critic_learning_rate = 1e-2

    visualize(track, adhdp, population, False)
    adhdp.plot_critic_gradient(0,1, 0)
    adhdp.plot_actor_critic(0,1)
    plt.show()


if __name__ == "__main__":
    #pretrain()
    main()