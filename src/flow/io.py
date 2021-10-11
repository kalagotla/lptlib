import numpy as np


class FlowIO:
    """Module to read-in a flow file and output flow parameters\n
    \n
    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com\n
    date: 10-05/2021\n
    \n
    Example:
        grid = FlowIO('plate.sp.x')  # Assume file is in the path\n
        flow.read_flow()  # Call method to read the data\n
        print(flow)  # prints the docstring for grid\n
        # Instance attributes\n
        print(flow.q.shape)  # shape of the flow data\n
        print(flow.ng)  # Number of blocks
        """

    def __init__(self, filename):
        self.filename = filename
        self.ng = None
        self.ni, self.nj, self.nk = None, None, None
        self.mach = None
        self.alpha = None
        self.rey = None
        self.time = None
        self.q = []

    def __str__(self):
        doc = "This instance has the filename " + self.filename + "\n" + \
              "q attribute is of shape 5xgrid_shape, rows representing density, momentum[*3], energy\n" + \
              "For example, rho = flow.q(0,...), flow being the object"
        return doc

    def read_flow(self):
        """Reads in the flow file and changes the instance attributes\n
        \n
        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com\n
        credit: Paul Orkwis\n
        date: 10-05/2021\n
        """
        with open(self.filename, 'r') as data:
            self.ng = np.fromfile(data, dtype='i4', count=1)[0]

            # Should be looped for multiple blocks
            self.ni, self.nj, self.nk = np.fromfile(data, dtype='i4', count=3)

            self.mach, self.alpha, self.rey, self.time = np.fromfile(data, dtype='f4', count=4)

            # Read-in flow data
            self.q = np.zeros((self.ni, self.nj, self.nk, 5))

            for m in range(5):
                for k in range(self.nk):
                    for j in range(self.nj):
                        self.q[:, j, k, m] = np.fromfile(data, dtype='f4', count=self.ni)

            print('Flow data reading is successful for ' + self.filename)
