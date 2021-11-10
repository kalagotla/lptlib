# Integrate from given point to produce streamlines


class Integration:

    def __init__(self, grid, flow, idx, interp):
        self.grid = grid
        self.flow = flow
        self.idx = idx
        self.interp = interp
        self.point = None

    def compute(self, time_step=1e-6):
        if self.idx.info == 'Given point is not in the domain. The cell attribute will return "None"\n':
            return
        from src.function.variables import Variables
        q_interp = Variables(self.interp)
        q_interp.compute_velocity()
        self.point = self.idx.point + q_interp.velocity * time_step

        return self.point
