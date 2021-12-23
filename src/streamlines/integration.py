# Integrate from given point to produce streamlines
import numpy as np


class Integration:

    def __init__(self, interp):
        self.interp = interp
        self.point = None

    def compute(self, method='c-space', time_step=1e-6):
        from src.function.variables import Variables

        if self.interp.idx.info == 'Given point is not in the domain. The cell attribute will return "None" ' \
                                   'in search algorithm\n':
            return

        _J_inv = self.interp.idx.grid.m2[self.interp.idx.cell[0][0], self.interp.idx.cell[0][1], self.interp.idx.cell[0][2], :, :, self.interp.idx.block]

        q_interp = Variables(self.interp)
        q_interp.compute_velocity()
        p_velocity = q_interp.velocity.reshape(3)
        c_velocity = np.matmul(_J_inv, p_velocity)
        self.point = self.interp.idx.point + c_velocity * time_step

        if method == 'c-space':
            if not np.all([0, 0, 0] <= self.point) or not np.all(
                    self.point + 1 < [self.interp.idx.grid.ni[self.interp.idx.block],
                                      self.interp.idx.grid.nj[self.interp.idx.block],
                                      self.interp.idx.grid.nk[self.interp.idx.block]]):
                self.point = None

        return self.point
