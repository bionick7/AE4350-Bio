import numpy as np
from typing import Callable
import pyray as rl


def color_from_html(html: int) -> rl.Color:
    return rl.Color((html >> 16) & 0xFF, (html >> 8) & 0xFF, html & 0xFF, 0xFF)

BG = color_from_html(0x2C2B30)
FG = color_from_html(0xEAEBDA)
HIGHLIGHT = color_from_html(0xE3524D)

RAD2DEG = 180/3.1415926535
DEG2RAD = 1/RAD2DEG

COLOR_TYPE = type(BG)  # since python thinks rl.Color is a function smh


def rk4(f_dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray], 
        x: np.ndarray, u: np.ndarray, dt: float):
    k1 = f_dynamics(x, u)
    k2 = f_dynamics(x + k1*dt/2, u)
    k3 = f_dynamics(x + k2*dt/2, u)
    k4 = f_dynamics(x + k3*dt, u)
    return x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
