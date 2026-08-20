import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        from lptlib.io import GridIO, FlowIO

        grd = GridIO(filename='../data/plate_data/plate.sp.x')
        flw = FlowIO('../data/plate_data/sol-0000010.q')

        import lptlib

        grd = lptlib.GridIO(filename='../data/plate_data/plate.sp.x')
        flw = lptlib.FlowIO('../data/plate_data/sol-0000010.q')


if __name__ == '__main__':
    unittest.main()
