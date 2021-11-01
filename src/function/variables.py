# Calculates different vairables from plot3d data
# Equations can be found here: https://www.grc.nasa.gov/WWW/winddocs/towne/plotc/plotc_p3d.html

import numpy as np


class Variables:

    def __init__(self, flow, gamma=1.4):
        self.flow = flow
        self.gamma = gamma
        self.velocity = None
        self.velocity_magnitude = None
        self.temperature = None
        self.pressure = None

    def compute_velocity(self):
        self.velocity = self.flow.q[..., 1:4] / (self.flow.q[..., 0, None] * self.flow.mach)
        self.velocity_magnitude = (self.velocity[..., 0]**2 + self.velocity[..., 1]**2 + self.velocity[..., 2]**2)**0.5

    def compute_temperature(self):
        self.compute_velocity()
        _R = 1 / (self.gamma * self.flow.mach**2)
        _E_T = self.flow.q[..., 4] / self.flow.mach**2
        self.temperature = (self.gamma - 1) * (_E_T/self.flow.q[..., 0] - self.velocity_magnitude/2) / _R

    def compute_pressure(self):
        self.compute_temperature()
        self.pressure = self.flow.q[..., 0] * self.temperature

    def compute(self):
        self.compute_pressure()  # implicitly runs compute_velocity and compute_temperature()
