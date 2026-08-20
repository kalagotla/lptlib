import unittest


class TestStreamlines(unittest.TestCase):
    def test_streamlines(self):
        from lptlib.streamlines import Streamlines
        from testdata import require_data
        sl = Streamlines(grid_file=require_data('plate_data', 'plate.sp.x'),
                         flow_file=require_data('plate_data', 'sol-0000010.q'),
                         point=[0.5, 0.5, 0.01], time_step=1)
        sl.compute()

    def test_streamlines_mb(self):
        from lptlib.streamlines import Streamlines
        from testdata import require_data
        sl = Streamlines(grid_file=require_data('multi_block', 'plate', 'plate.mb.sp.x'),
                         flow_file=require_data('multi_block', 'plate', 'plate.mb.sp.q'),
                         point=[0.5, 0.5, 0.01], time_step=1)
        sl.compute()


if __name__ == '__main__':
    unittest.main()
