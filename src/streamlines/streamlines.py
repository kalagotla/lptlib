# Uses the program API to extract streamlines

class Streamlines:
    def __init__(self, grid_file, flow_file, point,
                 search_method='p-space', interpolation_method='p-space', integration_method='p-space'):
        self.grid_file = grid_file
        self.flow_file = flow_file
        self.point = point
        self.search_method = search_method
        self.interpolation_method = interpolation_method
        self.integration_method = integration_method
        self.streamline = point

    def compute(self):
        from src.function.timer import Timer
        from src.io.plot3dio import GridIO
        from src.io.plot3dio import FlowIO
        from src.streamlines.search import Search
        from src.streamlines.interpolation import Interpolation
        from src.streamlines.integration import Integration

        grid = GridIO(self.grid_file)
        flow = FlowIO(self.flow_file)

        # Read in the grid and flow data
        grid.read_grid()
        flow.read_flow()
        grid.compute_metrics()

        while True:
            with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                idx = Search(grid, point)
                interp = Interpolation(flow, idx)
                intg = Integration(interp)
                idx.compute(method=self.search_method)
                interp.compute(method=self.interpolation_method)
                new_point = intg.compute(method=self.integration_method, time_step=1)
                if new_point is None:
                    print('Integration complete!')
                    break
                self.streamline.append(new_point)
                point = new_point

        return self.streamline
