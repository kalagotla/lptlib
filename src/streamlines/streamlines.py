# Uses the program API to extract streamlines
import numpy as np


class Streamlines:
    """
    Module to extract particle path information

    ...

    Attributes
    ----------
    Input:
        grid_file : str
            Path to the plot3d grid data file
        flow_file : str
            Path to the plot3d flow data file
        point : list
            Starting point for integration
        search_method : str
            Default is p-space; can specify c-space
        interpolation_method : str
            Default is p-space; can specify c-space
        integration_method : str
            Default is p-space; can specify c-space
        time_step : float
            Default is 1e-3

    Output:
        streamline : numpy.ndarray
            shape is nx3; each column represents x, y, z

    Methods
    -------
    compute()
        integrates and returns the streamline ndarray
    ...

    Example:
        sl = Streamlines('../../data/vortex/vortex.sb.sp.x', '../../data/vortex/vortex.sb.sp.q', [-0.05, 0.05, 5])
        sl.compute()
    -------


    """
    def __init__(self, grid_file, flow_file, point,
                 search_method='p-space', interpolation_method='p-space', integration_method='p-space',
                 time_step=1e-3):
        self.grid_file = grid_file
        self.flow_file = flow_file
        self.point = np.array(point)
        self.search_method = search_method
        self.interpolation_method = interpolation_method
        self.integration_method = integration_method
        self.time_step = time_step
        self.streamline = []

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

        self.streamline.append(self.point)
        while True:
            with Timer(text="Elapsed time for loop number " + str(len(self.streamline)) + ": {:.8f}"):
                idx = Search(grid, self.point)
                interp = Interpolation(flow, idx)
                intg = Integration(interp)
                idx.compute(method=self.search_method)
                interp.compute(method=self.interpolation_method)
                new_point = intg.compute(method=self.integration_method, time_step=self.time_step)
                if new_point is None:
                    print('Integration complete!')
                    break
                self.streamline.append(new_point)
                self.point = new_point

        return
