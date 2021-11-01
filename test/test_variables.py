import unittest


class TestVariables(unittest.TestCase):
    from src.io.plot3dio import GridIO, FlowIO
    from src.streamlines.search import Search
    from src.streamlines.interpolation import Interpolation

    grid = GridIO('../data/plate_data/plate.sp.x')
    flow = FlowIO('../data/plate_data/sol-0000010.q')
    idx = Search(grid, [8.5, 0.5, 0.01])
    point_data = Interpolation(grid, flow, idx)

    grid.read_grid()
    flow.read_flow()
    idx.compute()
    point_data.compute()

    def test_variables(self, flow=flow):
        from src.function.variables import Variables

        variables = Variables(flow)
        variables.compute_velocity()
        variables.compute_temperature()

        self.assertEqual(variables.velocity.shape, (720, 152, 129, 3))
        self.assertEqual(variables.temperature.shape, (720, 152, 129))

    def test_variables_compute(self, flow=flow):
        from src.function.variables import Variables

        variables = Variables(flow)
        variables.compute()

        self.assertEqual(variables.velocity.shape, (720, 152, 129, 3))
        self.assertEqual(variables.temperature.shape, (720, 152, 129))

    def _test(self, point_variables):
        """
        Inside function to test point_variables

        :param point_variables: src.function.variables.Variables
        :return: None
        """

        self.assertEqual(
            sum(abs(point_variables.velocity - [1.02376411e+00, -5.38663727e-05, 6.40703744e-08])) <= 1e-6,
            True)
        self.assertEqual(abs(point_variables.temperature - 1.0003078917225714) <= 1e-6, True)

    def test_point_variables(self, point_data=point_data):
        from src.function.variables import Variables

        point_variables = Variables(point_data)
        point_variables.compute_velocity()
        point_variables.compute_temperature()

        self._test(point_variables)

    def test_point_variables_compute(self, point_data=point_data):
        from src.function.variables import Variables

        point_variables = Variables(point_data)
        point_variables.compute()

        self._test(point_variables)


if __name__ == '__main__':
    unittest.main()
