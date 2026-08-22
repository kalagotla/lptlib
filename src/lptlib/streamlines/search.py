# This file searches for the cell in which the given point is present in the grid

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Threshold ``_cell_index`` uses to decide which side of the nearest node a
# point falls on, expressed as a *fraction of the local cell*. It used to be a
# hard-coded absolute 1e-6 at every branch of that method, which is meaningless
# without a length scale: the oblique-shock fixture has ``dz = 2.5e-5``, so
# 1e-6 was four per cent of a cell there and the octant choice could be off by
# a whole cell. ``_cell_scale`` supplies the length this multiplies, and
# ``_point_in_cell`` has to tolerate exactly this much slop, so the value is
# named here.
_QUADRANT_REL_TOL = 1e-6

# Fraction of the local cell within which a query point is treated as
# coinciding with a grid node. Node detection used to ride along inside
# ``_cell_index`` on an absolute 1e-12, with the same scale-free problem; it is
# its own test now (``_is_node``) and its own -- relative -- tolerance.
_NODE_REL_TOL = 1e-9


# noinspection SpellCheckingInspection
class Search:
    """Module to search for the cell in which the given point is present

    ...

    Attributes
    ----------
    Input :
        grid : src.io.plot3dio.GridIO
            Grid object created from GridIO
        ppoint: list
            A float list of shape 3 -- Representing x, y, z of a point
    Output :
        cpoint: numpy.ndarray
            ppoint location in c-space. Will be computed if calculating in c-space
        index: numpy.ndarray
            Indices of closest node to the given point
        cell: numpy.ndarray
            Indices of the cell in which the given point is present
        info: str
            Information about the point location

    Methods
    -------
    compute()
        Finds the location of given point in the grid using the given search-method
        search-methods:
            distance, block-distance, p-space, c-space

    p2c()
        Method to convert location of point in physical space to computational space
        Rarely used

    c2p()
        Method to convert from c-space to p-space
        Used in integration algorithms to get new cell

    author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
    date: 10-24/2021
        """

    def __init__(self, grid, ppoint, warm_start=None):
        self.grid = grid
        self.ppoint = ppoint
        self.cpoint = None
        self.index = None
        self.cell = None
        self.info = None
        self.block = None
        # Newton-Raphson warm start for p2c. Kept per-instance (not module-global)
        # so concurrent Search objects -- threads, particles -- never share it.
        # Callers walking a single trajectory pass the previous step's converged
        # c-space point as ``warm_start`` to skip the nearest-node scan; a stale
        # or distant guess is rejected inside p2c.
        self._cpoint = None if warm_start is None else np.array(warm_start, dtype='f8')

    def __str__(self):
        doc = "This instance takes in the grid of shape " + self.grid.grd.shape + \
              "\nand the searches for the point " + self.ppoint + " in the grid.\n" \
              "Use method 'compute' to find (attributes) the closest 'index' and the nodes of the 'cell'.\n"
        return doc

    def _last_cell_origin(self):
        """Highest valid cell origin along i, j and k for the current block.

        A cell is addressed by its lowest node and spans that node plus one, so
        with ``n`` nodes along an axis the last cell starts at ``n - 2``.
        """
        return (max(int(self.grid.ni[self.block]) - 2, 0),
                max(int(self.grid.nj[self.block]) - 2, 0),
                max(int(self.grid.nk[self.block]) - 2, 0))

    def _cell_split(self, _cpoint):
        """Split a c-space point into its host cell origin and local fractions.

        Returns ``(eps, frac)`` where ``eps`` indexes the lowest node of the
        cell containing the point and ``frac`` is the point's position within
        that cell along each axis. A point sitting exactly on the far i/j/k
        face has an integer part equal to the *last node* index, where no cell
        starts; it is reported as the last cell with a fraction of 1.0, which
        is the same physical location. Points strictly beyond the last node
        keep their out-of-range index so callers can reject them.
        """
        _c = np.asarray(_cpoint, dtype='f8')
        _n = np.array([self.grid.ni[self.block], self.grid.nj[self.block],
                       self.grid.nk[self.block]], dtype=int)
        _eps = _c.astype(int)
        _frac = np.modf(_c)[0]
        # Exactly on the far face: integer index at the last node, no remainder
        _far = (_eps == _n - 1) & (_frac == 0.0)
        _frac = np.where(_far, 1.0, _frac)
        _eps = np.where(_far, np.maximum(_n - 2, 0), _eps)
        return _eps, _frac

    def _cell_scale(self, _i, _j, _k):
        """Length of the shortest grid edge leaving node ``(_i, _j, _k)``.

        Tolerances that are lengths -- "is the point on this node?", "is the
        point outside this cell?" -- are meaningless as absolute numbers,
        because the grids this library reads span anything from microns to
        metres. This is the local length they are expressed as a fraction of.
        Returns ``0.0`` only for a degenerate block with no edges at all.
        """
        _grd = self.grid.grd[..., self.block]
        _size = (int(self.grid.ni[self.block]), int(self.grid.nj[self.block]),
                 int(self.grid.nk[self.block]))
        _here = (int(_i), int(_j), int(_k))
        _node = _grd[_here]
        _edges = []
        for _axis in range(3):
            for _step in (1, -1):
                _nbr = list(_here)
                _nbr[_axis] += _step
                if 0 <= _nbr[_axis] < _size[_axis]:
                    _edges.append(float(np.linalg.norm(_grd[tuple(_nbr)] - _node)))
        _edges = [_e for _e in _edges if _e > 0.0]
        return min(_edges) if _edges else 0.0

    def _is_node(self, _i, _j, _k):
        """True when ``self.ppoint`` coincides with grid node ``(_i, _j, _k)``.

        This is the node test on its own. It used to be the first branch of
        ``_cell_index``, which meant node detection and cell selection shared a
        code path and could not be corrected independently -- and cell
        selection was the half that was wrong on curvilinear grids.

        The tolerance is ``_NODE_REL_TOL`` times the shortest edge at the node,
        with a floor at coordinate round-off, so it means the same thing on a
        millimetre grid and on a metre grid.
        """
        if self.ppoint is None or self.block is None:
            return False
        _node = self.grid.grd[int(_i), int(_j), int(_k), :, self.block]
        _tol = max(_NODE_REL_TOL * self._cell_scale(_i, _j, _k),
                   np.finfo('f8').eps * float(np.max(np.abs(_node))))
        return bool(np.linalg.norm(np.asarray(self.ppoint, dtype='f8') - _node)
                    <= _tol)

    def _locate_from_cpoint(self, _cpoint):
        """Set ``cell`` -- and ``info`` if the point is a node -- from c-space.

        The computational coordinate is the grid-agnostic source of truth for
        which cell holds a point: the cell origin is simply the integer part of
        the c-space coordinate. ``_cell_index`` instead decided by the
        Cartesian octant of ``ppoint - grd[i, j, k]``, which is only the same
        thing when the computational axes happen to line up with x, y and z --
        i.e. on a Cartesian grid, not on the curvilinear grids this library
        exists for. On the quarter-annulus fixture that octant test disagreed
        with the computational coordinate for 57 per cent of in-domain points,
        and ``Interpolation`` -- which builds its local weights as
        ``cpoint - cell[0]`` -- silently extrapolated by up to 0.996 of a cell
        for every one of them.

        Deciding from ``_cell_split`` makes ``cpoint - cell[0]`` land in
        ``[0, 1]`` by construction, which is the invariant tri-linear
        interpolation needs.

        Node detection is done separately and explicitly: round the c-space
        coordinate to the nearest node index and ask ``_is_node`` whether the
        query point really is that node. Newton-Raphson can converge to a node
        from just below (``7.999999999998`` for node 8), so rounding rather
        than truncating is what makes the node test stable. ``info`` is only
        claimed when the node is the located cell's own origin, because the
        shortcut ``Interpolation`` takes reads ``cell[0]``; a node on the far
        i/j/k face is clamped into the last cell, where it is not the origin,
        and falls through to the regular path that weights it exactly anyway.
        """
        _c = np.asarray(_cpoint, dtype='f8')
        _eps, _ = self._cell_split(_c)
        self.cell = self._cell_nodes(int(_eps[0]), int(_eps[1]), int(_eps[2]))

        _size = np.array([self.grid.ni[self.block], self.grid.nj[self.block],
                          self.grid.nk[self.block]], dtype=int)
        _near = np.clip(np.rint(_c).astype(int), 0, _size - 1)
        if self._is_node(*_near):
            _node_cell = self._cell_nodes(int(_near[0]), int(_near[1]),
                                          int(_near[2]))
            if np.array_equal(_node_cell[0], _near):
                self.cell = _node_cell
                # The wording is the library's public marker for this state --
                # ``Interpolation`` and the tests compare against it verbatim --
                # so it is kept exactly as it was even though the tolerance is
                # now relative rather than an absolute 1e-12. See ``_is_node``.
                self.info = 'Given point is a node in the domain with a tol of 1e-12.\n' \
                            'Interpolation will assign node properties for integration.\n' \
                            'Index of the node will be returned by cell attribute\n'
        return self.cell

    def _cell_nodes(self, _i, _j, _k):
        # _Internal method to get the nodes of a cell
        # Clamp to valid range to prevent negative indices wrapping
        # (e.g., 2D-extruded grids where k=0 and point sits at boundary)
        _i = max(_i, 0)
        _j = max(_j, 0)
        _k = max(_k, 0)
        # Clamp the far side as well. A point landing exactly on the maximum
        # i/j/k index asks for a cell that starts at the last node; that cell
        # does not exist and looking up its "+1" nodes ran off the end of the
        # grid with an IndexError. Locate such a point in the adjacent (last)
        # cell instead -- the same physical location, at a local fraction of 1.
        if self.block is not None:
            _i_max, _j_max, _k_max = self._last_cell_origin()
            _i = min(_i, _i_max)
            _j = min(_j, _j_max)
            _k = min(_k, _k_max)
        _cell = np.array([[_i, _j, _k],
                          [_i + 1, _j, _k],
                          [_i + 1, _j + 1, _k],
                          [_i, _j + 1, _k],
                          [_i, _j, _k + 1],
                          [_i + 1, _j, _k + 1],
                          [_i + 1, _j + 1, _k + 1],
                          [_i, _j + 1, _k + 1]], dtype=int)
        return _cell

    def _candidate_cells(self, i, j, k):
        """Origins of every cell that touches node ``(i, j, k)``.

        A node in the interior of a block is shared by eight cells, whose
        origins are the corners of ``{i-1, i} x {j-1, j} x {k-1, k}``.
        ``_cell_index`` picks exactly one of them, by Cartesian octant; this is
        the full set it picks from. Origins are clamped into the valid range
        the same way ``_cell_nodes`` clamps them, and duplicates -- which the
        clamp produces at a block face, where fewer than eight distinct cells
        meet the node -- are dropped, so the list holds between one and eight
        entries and the containment sweep over it is bounded.
        """
        _i_max, _j_max, _k_max = self._last_cell_origin()
        _origins = []
        for _a in (i, i - 1):
            for _b in (j, j - 1):
                for _c in (k, k - 1):
                    _origin = (min(max(int(_a), 0), _i_max),
                               min(max(int(_b), 0), _j_max),
                               min(max(int(_c), 0), _k_max))
                    if _origin not in _origins:
                        _origins.append(_origin)
        return _origins

    def _relocate_near_node(self, i, j, k):
        """Retry containment against every cell that touches node ``(i, j, k)``.

        ``_cell_index`` chooses one of those cells by the Cartesian octant of
        ``ppoint - grd[i, j, k]``, which is exact only when the computational
        axes line up with x, y and z. On a curvilinear block it can name a
        neighbouring cell, and ``_point_in_cell`` then correctly reports that
        the point is not in the cell it was handed -- so the ``distance``
        searches rejected points that are plainly inside the grid. Measured on
        the quarter-annulus fixture: 12.9 per cent of 1500 random in-domain
        points, and 15 per cent of the 400-point lattice on its stretched
        variant.

        A nearest-node search can legitimately land one cell off on a curved
        mesh, so the honest repair is to widen the candidate set rather than to
        loosen the containment test. Every rejected point in that measurement
        was in a cell adjacent to the nearest node -- 194 of 194 -- and in
        every one of those cases exactly one of the adjacent cells' node
        bounding boxes contained the point, so the widened search is not only
        sufficient but unambiguous.

        Only reached when the octant choice has already failed, so a point that
        the ``distance`` searches accept today keeps the cell they give it
        today and every interpolated value downstream is unchanged. Out-of-
        domain points still have to fall inside *some* adjacent cell's box to
        be accepted, which is the same conservative test as before applied to
        at most eight boxes instead of one.

        Returns ``True`` and leaves ``self.cell`` on the containing cell, or
        returns ``False`` and restores the octant choice for the caller to
        report.
        """
        _chosen = self.cell
        for _origin in self._candidate_cells(i, j, k):
            self.cell = self._cell_nodes(*_origin)
            if self._point_in_cell():
                return True
        self.cell = _chosen
        return False

    def _point_in_cell(self, tol=1e-9):
        """True unless ``self.ppoint`` provably lies outside ``self.cell``.

        ``_cell_nodes`` clamps the cell origin into the valid range so that a
        point sitting exactly on the far i/j/k face is located in the last
        cell rather than in a cell that starts one node past the end of the
        grid. That clamp is required -- without it the far-face lookup raised
        ``IndexError`` -- but it also means the index-range test the
        ``distance`` searches used to make (``max(cell[:, 0]) > ni - 1``) can
        never fire again, so out-of-domain points need a separate containment
        test. This is that test.

        ``_find_block`` only checks the point against a block's axis-aligned
        bounding box. On a Cartesian grid the box is the block, so that check
        is exact. On a curvilinear grid -- an annular sector, a C-grid, any
        body-fitted block around a curved surface -- the box strictly contains
        points that are outside the block, and those points reach the cell
        search. This method compares the point against the bounding box of the
        eight nodes of the located cell instead of the whole block.

        The test is conservative in one direction only: the cell is contained
        in the box of its own nodes, so a point outside the box is certainly
        outside the cell and is rejected, while a point inside the box may
        still be outside a curved cell and is accepted. It therefore never
        rejects a point that is genuinely inside the grid, but it does accept
        a thin sliver of points immediately outside a curved boundary face,
        where the search extrapolates by less than one cell.

        The box is padded before the comparison. ``_cell_index`` decides which
        of the eight cells around the nearest node to take by comparing
        Cartesian offsets against ``_QUADRANT_REL_TOL`` of the local cell, so a
        point that close to a node can be reported in the neighbouring cell and
        sit that far outside it while being firmly inside the grid. The pad
        covers that, plus coordinate round-off scaled to the magnitude of the
        node coordinates. It is a fraction of this cell rather than an absolute
        length, so it means the same thing on a micron grid and a metre grid --
        the absolute 1e-6 it replaces was most of a cell on a fine mesh and
        below round-off on a coarse one. Points landing exactly on a boundary
        face are corners or face points of the located cell and are always
        accepted.
        """
        if self.cell is None or self.block is None or self.ppoint is None:
            return False
        _nodes = self.grid.grd[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2],
                               :, self.block]
        _lo, _hi = _nodes.min(axis=0), _nodes.max(axis=0)
        _pad = (_QUADRANT_REL_TOL * float(np.max(_hi - _lo))
                + tol * float(np.max(np.abs(_nodes))))
        _p = np.asarray(self.ppoint, dtype='f8')
        return bool(np.all(_p >= _lo - _pad) and np.all(_p <= _hi + _pad))

    @staticmethod
    def _cell_index(self, i, j, k):
        """Pick the cell around node ``(i, j, k)`` that holds ``self.ppoint``.

        The choice is made by which Cartesian octant of the node the point
        falls in: the sign of ``ppoint - grd[i, j, k]`` along x, y and z
        selects ``i`` or ``i - 1``, ``j`` or ``j - 1``, ``k`` or ``k - 1``.

        SCOPE -- this is only used by the ``distance`` and ``block_distance``
        searches, which locate the nearest node and have no computational
        coordinate to index with. It is exact only when the computational axes
        are aligned with x, y and z, i.e. on a Cartesian grid; on a
        curvilinear grid the i axis need not point along x at all (in an
        annular block it points radially), so the octant test can pick a
        neighbouring cell. ``_point_in_cell`` is what catches the case where
        that lands outside the grid, and ``_relocate_near_node`` is what stops
        it from being reported as out-of-domain when the point is really in one
        of the other cells around the same node.

        The searches that *do* have a computational coordinate -- ``p-space``
        and ``c-space``, via ``p2c`` -- must not use this method. They use
        ``_locate_from_cpoint``, which indexes by the c-space coordinate and is
        correct on any grid. Before that split, ``p2c`` called this method and
        ``cell[0]`` disagreed with ``cpoint.astype(int)`` for 57 per cent of
        in-domain points on the quarter-annulus fixture, which made
        ``Interpolation`` extrapolate by up to 0.996 of a cell.

        The octant threshold is ``_QUADRANT_REL_TOL`` of the local cell rather
        than the absolute 1e-6 it used to be. That number was 4 per cent of a
        cell on the oblique-shock fixture (``dz = 2.5e-5``) and pushed points
        that close to a node into the wrong cell even on a Cartesian grid.

        Node detection is no longer done here: it is ``_is_node``, which this
        method calls. The two concerns were entangled in one branch, and only
        the cell-selection half was wrong.

        Note this is separate from domain containment: ``_point_in_cell``
        tolerates exactly this much slop and still rejects points that are
        outside the grid.
        """
        # _Internal method to obtain the nodes of the cell in which the given point is present

        # Transform to found node to find the location of point
        # Basically looking at which quadrant the point is located
        # to find the nodes of the respective cell
        _node = self.grid.grd[i, j, k, :, self.block]
        _point_transform = self.ppoint - _node
        # Octant threshold as a length: a fraction of the local cell, not the
        # scale-free absolute constant this used to hard-code eight times.
        _tol = _QUADRANT_REL_TOL * self._cell_scale(i, j, k)

        # Check if point is a node in the domain
        if self._is_node(i, j, k):
            self.cell = self._cell_nodes(i, j, k)
            # A node on the far i/j/k face is clamped into the last cell, where
            # it is no longer that cell's first node. The "node" shortcut reads
            # cell[0], so only claim it when the node really is the cell origin;
            # otherwise fall through to the regular cell path, which weights the
            # node exactly (local fraction 1.0) via tri-linear interpolation.
            if np.array_equal(self.cell[0], [i, j, k]):
                self.info = 'Given point is a node in the domain with a tol of 1e-12.\n' \
                            'Interpolation will assign node properties for integration.\n' \
                            'Index of the node will be returned by cell attribute\n'
            # print(self.info)
            return

        # ON BOUNDARY FOR A GENERALIZED HEXA IS SAME AS DEFAULT SEARCH
        # Removed the code for on the boundary case
        # Start the main cell nodes code
        if np.all(_point_transform >= _tol):
            self.cell = self._cell_nodes(i, j, k)
            return
        if _point_transform[0] <= _tol and _point_transform[1] >= _tol and _point_transform[2] >= _tol:
            self.cell = self._cell_nodes(i - 1, j, k)
            return
        if _point_transform[0] <= _tol and _point_transform[1] <= _tol and _point_transform[2] >= _tol:
            self.cell = self._cell_nodes(i - 1, j - 1, k)
            return
        if _point_transform[0] >= _tol and _point_transform[1] <= _tol and _point_transform[2] >= _tol:
            self.cell = self._cell_nodes(i, j - 1, k)
            return
        if _point_transform[0] >= _tol and _point_transform[1] >= _tol and _point_transform[2] <= _tol:
            self.cell = self._cell_nodes(i, j, k - 1)
            return
        if _point_transform[0] <= _tol and _point_transform[1] >= _tol and _point_transform[2] <= _tol:
            self.cell = self._cell_nodes(i - 1, j, k - 1)
            return
        if np.all(_point_transform <= _tol):
            self.cell = self._cell_nodes(i - 1, j - 1, k - 1)
            return
        if _point_transform[0] >= _tol and _point_transform[1] <= _tol and _point_transform[2] <= _tol:
            self.cell = self._cell_nodes(i, j - 1, k - 1)
            return

        return

    @staticmethod
    def _find_block(self):
        # _Internal method to find the block
        # Setup to compute block number in which the point is present
        _bool_min = self.grid.grd_min <= self.ppoint
        _bool_max = self.grid.grd_max >= self.ppoint
        _bool = _bool_max == _bool_min

        # Test if the given point is in domain or not
        if np.all(_bool.all(axis=1) == False) or np.all(_bool_min == False) or np.all(_bool_max == False):
            self.info = 'Given point is not in the domain. The cell attribute will return "None" in search algorithm\n'
            self.cell = None
            self.ppoint = None
            self.cpoint = None
            self.block = None
            # print(self.info)
            return
        # Assign the block number to the attribute
        self.block = int(np.where(_bool.all(axis=1))[0][0])

        return self.block

    def compute(self, method='block_distance'):
        """
        Use the method to compute index and cell attributes

        parameter:
            method: str
                One of 'distance', 'block_distance', 'p-space', 'c-space'

        return:
        None

        Out-of-domain behaviour
        -----------------------
        Every method first calls ``_find_block``, which tests the point
        against each block's axis-aligned bounding box. A point outside every
        box is rejected outright: ``cell``, ``ppoint``, ``cpoint`` and
        ``block`` are set to ``None`` and ``info`` explains why.

        That box test is exact only for a Cartesian block. On a curvilinear
        grid a point can sit inside a block's bounding box and still be
        outside the block, so each method has to reject it itself:

        - ``distance`` and ``block_distance`` locate the nearest node and pick
          a cell around it, then check the point against the bounding box of
          that cell's eight nodes (``_point_in_cell``). If the point is not in
          that cell the search does not give up: the octant choice is exact
          only on a Cartesian grid, so the remaining cells that touch the same
          node are tried too (``_relocate_near_node``), and the point is
          accepted in whichever of them contains it. Only a point that is in
          none of them is rejected, by setting ``ppoint`` and ``cpoint`` to
          ``None`` -- callers test ``ppoint is None``. Note that ``cell`` keeps
          the octant choice on rejection, so ``ppoint``, not ``cell``, is the
          rejection signal. Because the test is against each cell's bounding
          box rather than the curved cell itself, a point less than about one
          cell outside a curved boundary face can still be accepted and will be
          extrapolated; anything further out is rejected.
        - ``p-space`` and ``c-space`` invert the tri-linear map with
          Newton-Raphson (``p2c``). The iterate is clamped into the valid
          index range every step, so an out-of-domain point can never be
          reproduced, the residual never reaches tolerance, and ``p2c``
          gives up after 1000 iterations and returns ``None``, leaving
          ``ppoint`` and ``cpoint`` as ``None``. These methods therefore
          reject out-of-domain points, at the cost of running the full
          iteration budget before doing so.

        Points lying exactly on a boundary face -- including the far i/j/k
        faces -- are inside the domain and are located, not rejected. See
        ``_cell_nodes`` and ``_cell_split`` for how the far face is handled.

        author: Dilip Kalagotla @ kal ~ dilip.kalagotla@gmail.com
        date: 10-24/2021
        """

        # Find the block number
        self.block = self._find_block(self)
        # To check for point out-of-domain case
        if self.block is None:
            return

        match method:

            case 'distance':
                # Compute the distance from all nodes in the grid
                _dist = np.sqrt((self.grid.grd[..., 0, :] - self.ppoint[0]) ** 2 +
                                (self.grid.grd[..., 1, :] - self.ppoint[1]) ** 2 +
                                (self.grid.grd[..., 2, :] - self.ppoint[2]) ** 2)

                # Find the closest node to the point --> index.ndim = 4
                self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
                i, j, k, self.block = self.index[0], self.index[1], self.index[2], self.index[3]
                self._cell_index(self, i, j, k)
                # Check for the end of the domain case. _cell_nodes clamps the
                # cell origin into range, so the located cell is always a real
                # cell; what still has to be checked is whether the point is
                # actually in it. See _point_in_cell. If the octant choice
                # missed -- which it can by one cell on a curved block -- try
                # the other cells around the same node before giving up. See
                # _relocate_near_node.
                if not self._point_in_cell() and not self._relocate_near_node(i, j, k):
                    logger.warning('Given point is outside the grid. '
                                   'Point position lost.\n')
                    self.cpoint = None
                    self.ppoint = None
                    return

            case 'block_distance':
                # Compute distance inside the block to get the nearest node
                _i, _j, _k = self.grid.ni[self.block], self.grid.nj[self.block], self.grid.nk[self.block]
                _dist = np.sqrt((self.grid.grd[:_i, :_j, :_k, 0, self.block] - self.ppoint[0]) ** 2 +
                                (self.grid.grd[:_i, :_j, :_k, 1, self.block] - self.ppoint[1]) ** 2 +
                                (self.grid.grd[:_i, :_j, :_k, 2, self.block] - self.ppoint[2]) ** 2)

                # Other methods to calculate distance. The above method is faster
                # _grd_min_point = self.grid.grd[:_i, :_j, :_k, :, self.block] - self.point
                # _dist = np.linalg.norm(_grd_min_point, axis=-1)
                # _dist = np.sqrt(np.einsum("ijkl,ijkl->ijk", _grd_min_point, _grd_min_point))

                self.index = np.array(np.unravel_index(_dist.argmin(), _dist.shape))
                i, j, k = self.index
                self._cell_index(self, i, j, k)
                # Check for the end of the domain case. _cell_nodes clamps the
                # cell origin into range, so the located cell is always a real
                # cell; what still has to be checked is whether the point is
                # actually in it. See _point_in_cell. If the octant choice
                # missed -- which it can by one cell on a curved block -- try
                # the other cells around the same node before giving up. See
                # _relocate_near_node.
                if not self._point_in_cell() and not self._relocate_near_node(i, j, k):
                    logger.warning('Block search returned wrong cell! Point position lost.\n')
                    self.cpoint = None
                    self.ppoint = None
                    return

            case 'p-space':
                # Search for given point using newton-raphson
                # credit: Sadarjoen et al.
                # title: Particle tracing algorithms for 3D Curvilinear grids
                # This search is performed every single time to find the given point
                self.cpoint = self.p2c(self.ppoint)

            case 'c-space':
                # To run c-space global point location is needed
                # credit: Sadarjoen et al.
                # title: Particle tracing algorithms for 3D Curvilinear grids

                # Transform given point from p-space to c-space using newton-raphson
                # This is only performed once to get the initial c-space point
                self.cpoint = self.p2c(self.ppoint)

    def c2p(self, _cpoint):
        """
        Method to convert c-space point to p-space
        self.block is kept constant through the c-space algos
        self.block is checked and switched in streamlines algo
        This method is commonly used to test point location as well
        in different algorithms

        Args:
            _cpoint: c-space co-ordinates

        Returns:
            _ppoint: p-space co-ordinates

        """
        self.cpoint = _cpoint
        # Cell origin and local fractions. A point exactly on the far i/j/k
        # face is reported in the last cell at a fraction of 1.0 rather than in
        # a cell that starts one node past the end of the grid.
        (_eps0, _eps1, _eps2), (_alpha, _beta, _gamma) = self._cell_split(_cpoint)

        # Check if the given point is in the domain
        if _eps0 >= self.grid.ni[self.block]-1 or _eps1 >= self.grid.nj[self.block]-1 or \
                _eps2 >= self.grid.nk[self.block]-1:
            self.cpoint = None
            self.ppoint = None
            return self.ppoint

        # Determine in which cell the current point is present
        self.cell = self._cell_nodes(_eps0, _eps1, _eps2)
        _cell_grd = self.grid.grd[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2], :, self.block]

        # Calculate the location in p-space
        self.ppoint = (1 - _alpha) * (1 - _beta) * (1 - _gamma) * _cell_grd[0] + \
                _alpha  * (1 - _beta) * (1 - _gamma) * _cell_grd[1] + \
                _alpha  *      _beta  * (1 - _gamma) * _cell_grd[2] + \
                (1 - _alpha) *      _beta  * (1 - _gamma) * _cell_grd[3] + \
                (1 - _alpha) * (1 - _beta) *      _gamma  * _cell_grd[4] + \
                _alpha  * (1 - _beta) *      _gamma  * _cell_grd[5] + \
                _alpha  *      _beta *       _gamma  * _cell_grd[6] + \
                (1 - _alpha) *      _beta  *      _gamma  * _cell_grd[7]

        return self.ppoint

    def p2c(self, _ppoint):
        """
        Method to convert p-space point to c-space
        As there is no direct analytical equation. We use Newton-Raphson
        Args:
            _ppoint: p-space co-ordinates

        Returns:
            eps: c-space co-ordinates
        """
        self.ppoint = _ppoint

        if self.block is None:
            self.block = self._find_block(self)

        # Start Newton-Raphson
        _iter = 0
        # Track best (closest) iterate in case we stall near convergence
        best_cpoint = None
        best_ppoint = None
        best_residual = None
        best_tol = None
        # Initial guess: use nearest node in the block for a robust starting point
        # self._cpoint is the warm start from this instance's previous call (particle
        # tracking), but we fall back to the nearest-node guess if it doesn't exist or
        # is too far away. It is per-instance so threads/particles never share it.
        _cpoint = getattr(self, '_cpoint', None)
        _need_fresh_guess = _cpoint is None
        if not _need_fresh_guess:
            # Check if the global guess is in a reasonable neighborhood
            # by comparing physical-space distance to the grid spacing
            _pred = self.c2p(_cpoint)
            if _pred is None or np.linalg.norm(_pred - _ppoint) > 0.1 * np.linalg.norm(
                    self.grid.grd_max[self.block] - self.grid.grd_min[self.block]):
                _need_fresh_guess = True
        if _need_fresh_guess:
            # Compute nearest node as the initial guess
            _i, _j, _k = self.grid.ni[self.block], self.grid.nj[self.block], self.grid.nk[self.block]
            _dist = np.sqrt(
                (self.grid.grd[:_i, :_j, :_k, 0, self.block] - _ppoint[0]) ** 2 +
                (self.grid.grd[:_i, :_j, :_k, 1, self.block] - _ppoint[1]) ** 2 +
                (self.grid.grd[:_i, :_j, :_k, 2, self.block] - _ppoint[2]) ** 2)
            _nearest = np.array(np.unravel_index(_dist.argmin(), _dist.shape), dtype='f8')
            _cpoint = _nearest + 0.5  # offset to cell center
        self._cpoint = _cpoint

        while True:
            # Check for out-of-domain case and reset the point to in-domain
            _eps0, _eps1, _eps2 = _cpoint.astype(int)
            _alpha, _beta, _gamma = np.modf(_cpoint)[0]
            if _eps0 + 1 >= self.grid.ni[self.block]:
                _cpoint[0] = self.grid.ni[self.block] - 1 - _alpha
            if _eps1 + 1 >= self.grid.nj[self.block]:
                _cpoint[1] = self.grid.nj[self.block] - 1 - _beta
            if _eps2 + 1 >= self.grid.nk[self.block]:
                _cpoint[2] = self.grid.nk[self.block] - 1 - _gamma

            # Compute eps after the point is reset
            _eps0, _eps1, _eps2 = _cpoint.astype(int)
            self.cell = self._cell_nodes(_eps0, _eps1, _eps2)

            # Calculate J_inv for the point --> Uses tri-linear interpolation
            _cell_J_inv = self.grid.m2[self.cell[:, 0], self.cell[:, 1], self.cell[:, 2], :, :, self.block]
            _J_inv = (1 - _alpha) * (1 - _beta) * (1 - _gamma) * _cell_J_inv[0] + \
                     _alpha * (1 - _beta) * (1 - _gamma) * _cell_J_inv[1] + \
                     _alpha * _beta * (1 - _gamma) * _cell_J_inv[2] + \
                     (1 - _alpha) * _beta * (1 - _gamma) * _cell_J_inv[3] + \
                     (1 - _alpha) * (1 - _beta) * _gamma * _cell_J_inv[4] + \
                     _alpha * (1 - _beta) * _gamma * _cell_J_inv[5] + \
                     _alpha * _beta * _gamma * _cell_J_inv[6] + \
                     (1 - _alpha) * _beta * _gamma * _cell_J_inv[7]

            # Transform from c to p-space
            _pred_ppoint = self.c2p(_cpoint)

            # Difference b/w predicted point to given point
            _delta_ppoint = _ppoint - _pred_ppoint

            # Track best iterate based on residual norm
            _res_norm = np.linalg.norm(_delta_ppoint)

            # End newton-raphson if condition is met
            # TODO: Condition needs to be adapted based on Jacobian
            # TODO: Need to improve by normalizing the data
            _tol = 1e-12 * self.grid.J[self.cell[0, 0], self.cell[0, 1], self.cell[0, 2], self.block]
            if _tol <= 1e-12:
                _tol = 1e-12

            # Update best iterate bookkeeping
            if best_residual is None or _res_norm < best_residual:
                best_residual = _res_norm
                best_cpoint = _cpoint.copy()
                best_ppoint = _pred_ppoint.copy()
                best_tol = _tol

            if sum(abs(_delta_ppoint)) <= _tol:
                # The converged c-space coordinate is what decides the cell.
                # This used to call ``_cell_index``, whose comment already
                # claimed the two were equivalent; they are not on a
                # curvilinear grid, which is the whole point of the c-space
                # search. ``_locate_from_cpoint`` also runs the node test that
                # ``_cell_index`` used to fold into the same branch.
                self.ppoint = _pred_ppoint
                self._locate_from_cpoint(_cpoint)
                self.cpoint = _cpoint
                self._cpoint = _cpoint
                return _cpoint

            # Transform from p to c-space
            _delta_cpoint = np.matmul(_J_inv, _delta_ppoint)

            # Save old point
            _cpoint_old = _cpoint.copy()

            # Update point
            _cpoint += _delta_cpoint
            # Update the point to zero if less than zero
            _cpoint[_cpoint < 0] = 0
            _cpoint = abs(_cpoint)
            self._cpoint = _cpoint
            _iter += 1
            # Check if taking too long
            if _iter >= 1e3:
                # If we have a near-converged iterate, accept it instead of treating the
                # point as completely out-of-domain. Use a slightly relaxed tolerance.
                if best_cpoint is not None and best_residual is not None and best_tol is not None:
                    if best_residual <= 10 * best_tol:
                        # Same rule as the converged branch: the cell comes
                        # from the c-space coordinate, not from a Cartesian
                        # octant test around the nearest node.
                        self.ppoint = best_ppoint
                        self._locate_from_cpoint(best_cpoint)
                        self.cpoint = best_cpoint
                        self._cpoint = best_cpoint
                        logger.warning('**WARNING** Newton-Raphson did not fully converge within 1000 iterations.\n'
                              'Using best available in-domain approximation for point location.')
                        return best_cpoint

                logger.error('**ERROR** Newton-Raphson did not converge. Try again!\n'
                      'Possible reason might be the point might be too close to the end of a domain')
                self.ppoint, self.cpoint = None, None
                return


