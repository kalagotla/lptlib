import unittest


class TestVariables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lptlib.io import GridIO, FlowIO
        from lptlib.streamlines import Search
        from lptlib.streamlines import Interpolation
        from testdata import require_data

        # Get the data at a given point
        cls.grid = GridIO(require_data('plate_data', 'plate.sp.x'))
        cls.flow = FlowIO(require_data('plate_data', 'sol-0000010.q'))
        cls.idx = Search(cls.grid, [8.5, 0.5, 0.01])
        cls.point_data = Interpolation(cls.flow, cls.idx)

        cls.grid.read_grid()
        cls.flow.read_flow()
        cls.idx.compute()
        cls.point_data.compute()

    def test_variables(self):
        """
        To test variables class for the whole domain
        Test each function separately
        :return: None
        """
        from lptlib import Variables

        variables = Variables(self.flow)
        variables.compute_velocity()
        variables.compute_temperature()

        self.assertEqual(variables.velocity.shape, (720, 152, 129, 3, 1))
        self.assertEqual(variables.temperature.shape, (720, 152, 129, 1))

    def test_variables_compute(self):
        """
        Test "compute" method in variables class
        :return: None
        """
        from lptlib import Variables

        variables = Variables(self.flow)
        variables.compute()

        self.assertEqual(variables.velocity.shape, (720, 152, 129, 3, 1))
        self.assertEqual(variables.temperature.shape, (720, 152, 129, 1))

    def _test(self, point_variables):
        """
        Inside function to test point_variables

        :param point_variables: Interpolation object
        :return: None
        """

        self.assertEqual(
            sum(abs(point_variables.velocity.reshape(3) - [1.02420611e-01, -5.38896289e-06, 6.40980361e-09])) <= 1e-6,
            True)
        self.assertEqual(abs(point_variables.temperature.reshape(1) - 0.97452141) <= 1e-6, True)

    def test_point_variables(self):
        """
        Test variables class for a single point
        :return: None
        """
        from lptlib import Variables

        point_variables = Variables(self.point_data)
        point_variables.compute_velocity()
        point_variables.compute_temperature()

        self._test(point_variables)

    def test_point_variables_compute(self):
        """
        Test "compute" method in variables class for a single point
        :return: None
        """
        from lptlib import Variables

        point_variables = Variables(self.point_data)
        point_variables.compute()

        self._test(point_variables)


if __name__ == '__main__':
    unittest.main()
