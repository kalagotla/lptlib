# Integrate from given point to produce streamlines
import numpy as np


class Integration:

    def __init__(self, interp):
        self.interp = interp
        self.point = None

    def compute(self, method='p-space', time_step=1e-6):
        from src.function.variables import Variables
        # TODO add switch case
        if method == 'p-space':
            # For p-space algos; the point -in-domain check was done in search
            if self.interp.idx.point is None:
                self.point = None
                return self.point

            q_interp = Variables(self.interp)
            q_interp.compute_velocity()
            self.point = self.interp.idx.point + q_interp.velocity.reshape(3) * time_step

            return self.point

        if method == 'c-space':
            _J_inv = self.interp.idx.grid.m2[self.interp.idx.cell[0][0], self.interp.idx.cell[0][1],
                                             self.interp.idx.cell[0][2], :, :, self.interp.idx.block]

            q_interp = Variables(self.interp)
            q_interp.compute_velocity()
            p_velocity = q_interp.velocity.reshape(3)
            c_velocity = np.matmul(_J_inv, p_velocity)
            self.point = self.interp.idx.point + c_velocity * time_step

            # For c-spce the point in-domain check is done after integration
            if not np.all([0, 0, 0] <= self.point) or not np.all(
                    self.point + 1 < [self.interp.idx.grid.ni[self.interp.idx.block],
                                      self.interp.idx.grid.nj[self.interp.idx.block],
                                      self.interp.idx.grid.nk[self.interp.idx.block]]):
                self.point = None
                return self.point

            return self.point

        if method == 'pRK4':
            from src.streamlines.interpolation import Interpolation
            from src.streamlines.search import Search

            # Start RK4 for p-space
            # For p-space algos; the point-in-domain check was done in search
            x0 = self.interp.idx.point
            if self.interp.idx.point is None:
                self.point = None
                return self.point
            q_interp = Variables(self.interp)
            q_interp.compute_velocity()
            v0 = q_interp.velocity.reshape(3)
            k0 = time_step * v0
            x1 = x0 + 0.5 * k0

            idx = Search(self.interp.idx.grid, x1)
            idx.compute()
            interp = Interpolation(self.interp.flow, idx)
            interp.compute()
            if interp.idx.point is None:
                self.point = None
                return self.point
            q_interp = Variables(interp)
            q_interp.compute_velocity()
            v1 = q_interp.velocity.reshape(3)
            k1 = time_step * v1
            x2 = x0 + 0.5 * k1

            idx = Search(self.interp.idx.grid, x2)
            idx.compute()
            interp = Interpolation(self.interp.flow, idx)
            interp.compute()
            if interp.idx.point is None:
                self.point = None
                return self.point
            q_interp = Variables(interp)
            q_interp.compute_velocity()
            v2 = q_interp.velocity.reshape(3)
            k2 = time_step * v2
            x3 = x0 + k2

            idx = Search(self.interp.idx.grid, x3)
            idx.compute()
            interp = Interpolation(self.interp.flow, idx)
            interp.compute()
            if interp.idx.point is None:
                self.point = None
                return self.point
            q_interp = Variables(interp)
            q_interp.compute_velocity()
            v3 = q_interp.velocity.reshape(3)
            k3 = time_step * v3
            x4 = x0 + 1/6 * (k0 + 2*k1 + 2*k2 + k3)

            self.point = x4

            return self.point

        if method == 'cRK4':
            from src.streamlines.interpolation import Interpolation
            from src.streamlines.search import Search

            x0 = self.interp.idx.point
            _J_inv = self.interp.J_inv

            q_interp = Variables(self.interp)
            q_interp.compute_velocity()
            p_velocity = q_interp.velocity.reshape(3)
            c_velocity = np.matmul(_J_inv, p_velocity)
            k0 = time_step * c_velocity
            x1 = x0 + 0.5 * k0
            # For c-space the point in-domain check is done after integration
            if not np.all([0, 0, 0] <= x1) or not np.all(
                    x1 + 1 < [self.interp.idx.grid.ni[self.interp.idx.block],
                              self.interp.idx.grid.nj[self.interp.idx.block],
                              self.interp.idx.grid.nk[self.interp.idx.block]]):
                self.point = None
                print('Cannot run integration. The given point is out of domain\n')
                return self.point

            idx = Search(self.interp.idx.grid, x1)
            idx.block = self.interp.idx.block
            idx.c2p(x1)  # This will change cell, point attributes
            interp = Interpolation(self.interp.flow, idx)
            interp.compute(method='c-space')
            _J_inv = interp.J_inv

            q_interp = Variables(interp)
            q_interp.compute_velocity()
            p_velocity = q_interp.velocity.reshape(3)
            c_velocity = np.matmul(_J_inv, p_velocity)
            k1 = time_step * c_velocity
            x2 = x0 + 0.5 * k1
            # For c-space the point in-domain check is done after integration
            if not np.all([0, 0, 0] <= x2) or not np.all(
                    x2 + 1 < [self.interp.idx.grid.ni[idx.block],
                              self.interp.idx.grid.nj[idx.block],
                              self.interp.idx.grid.nk[idx.block]]):
                self.point = None
                print('Cannot run integration. The given point is out of domain\n')
                return self.point

            idx = Search(self.interp.idx.grid, x2)
            idx.block = self.interp.idx.block
            idx.c2p(x2)  # This will change cell attribute
            interp = Interpolation(self.interp.flow, idx)
            interp.compute(method='c-space')
            _J_inv = interp.J_inv

            q_interp = Variables(interp)
            q_interp.compute_velocity()
            p_velocity = q_interp.velocity.reshape(3)
            c_velocity = np.matmul(_J_inv, p_velocity)
            k2 = time_step * c_velocity
            x3 = x0 + k2
            # For c-space the point in-domain check is done after integration
            if not np.all([0, 0, 0] <= x3) or not np.all(
                    x3 + 1 < [self.interp.idx.grid.ni[idx.block],
                              self.interp.idx.grid.nj[idx.block],
                              self.interp.idx.grid.nk[idx.block]]):
                self.point = None
                print('Cannot run integration. The given point is out of domain\n')
                return self.point

            idx = Search(self.interp.idx.grid, x3)
            idx.block = self.interp.idx.block
            idx.c2p(x3)  # This will change cell attribute
            interp = Interpolation(self.interp.flow, idx)
            interp.compute(method='c-space')
            _J_inv = interp.J_inv

            q_interp = Variables(interp)
            q_interp.compute_velocity()
            p_velocity = q_interp.velocity.reshape(3)
            c_velocity = np.matmul(_J_inv, p_velocity)
            k3 = time_step * c_velocity
            x4 = x0 + 1/6 * (k0 + 2*k1 + 2*k2 + k3)
            # For c-spce the point in-domain check is done after integration
            if not np.all([0, 0, 0] <= x4) or not np.all(
                    x4 + 1 < [self.interp.idx.grid.ni[idx.block],
                              self.interp.idx.grid.nj[idx.block],
                              self.interp.idx.grid.nk[idx.block]]):
                self.point = None
                print('Cannot run integration. The given point is out of domain\n')
                return self.point

            self.point = x4

            return self.point
