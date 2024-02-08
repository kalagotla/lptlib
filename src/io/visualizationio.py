#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mayavi import mlab
from tvtk.api import tvtk


# In[2]:


class Visualization:
    
    def __init__(self, grid):
        
        if grid.nb == 1:
            self.grd = grid.grd
            #self.sol = sol

            #from numpy import array
            try:
                self.engine = mayavi.engine
            except NameError:
                from mayavi.api import Engine
                self.engine = Engine()
                self.engine.start()
            if len(self.engine.scenes) == 0:
                self.engine.new_scene()

            scene = self.engine.scenes[0]
            scene.scene.show_axes = True

            self.x = self.grd[:,:,:,0]
            self.y = self.grd[:,:,:,1]
            self.z = self.grd[:,:,:,2]  

            self.sgrd = tvtk.StructuredGrid(dimensions=(self.x.shape[0], self.x.shape[1], self.x.shape[2]))
            self.sgrd.points = np.c_[self.x.T.ravel(),
                                     self.y.T.ravel(),
                                     self.z.T.ravel()]

            self.src = mlab.pipeline.add_dataset(self.sgrd)
            
        else:
            self.grid = grid.grd
            self.nb = grid.nb
            self.ni = grid.ni
            self.nj = grid.nj
            self.nk = grid.nk

            #from numpy import array
            try:
                self.engine = mayavi.engine
            except NameError:
                from mayavi.api import Engine
                self.engine = Engine()
                self.engine.start()
            if len(self.engine.scenes) == 0:
                self.engine.new_scene()

            scene = self.engine.scenes[0]
            scene.scene.show_axes = True
            
    def view(self, plane):
        #self.plane = plane
        scene = self.engine.scenes[0]
        if plane == 'z+':scene.scene.z_plus_view()
        elif plane == 'z-':scene.scene.z_minus_view()
        elif plane== 'x+':scene.scene.x_plus_view()
        elif plane == 'x-':scene.scene.x_minus_view()
        elif plane == 'y+':scene.scene.y_plus_view()
        else: scene.scene.y_minus_view()

    def outline(self):
        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.pipeline.outline(self.src)

    def grid_plane(self, axis='x', position=0):
        mlab.clf()
        self.axis = axis
        self.position = position

        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.pipeline.grid_plane(self.src)

        grid_plane = self.engine.scenes[0].children[0].children[0].children[0]
        grid_plane.grid_plane.axis = self.axis
        grid_plane.grid_plane.position = self.position
        scene = self.engine.scenes[0]
        if axis == 'x': scene.scene.x_plus_view()
        elif axis == 'y': scene.scene.y_plus_view()
        else: scene.scene.z_plus_view()

    def add_parameter(self, quantity, name):
        self.quantity = quantity
        self.name = name
        mlab.clf()
        #add = self.src.add_attribute(self.quantity.T.ravel(), self.name)
        self.sgrd.point_data.scalars = self.quantity.T.ravel()
        self.sgrd.point_data.scalars.name = self.name
        self.src = mlab.pipeline.add_dataset(self.sgrd)

    def scalar_plane(self, ax='z_axes'): #enabled=True, tubing=True):
        self.ax = ax

        mlab.clf()
        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.pipeline.scalar_cut_plane(self.src, plane_orientation=ax)
        
        #scalar_cut_plane = self.engine.scenes[0].children[0].children[0].children[0]
        #scalar_cut_plane.implicit_plane.widget.enabled = enabled
        #scalar_cut_plane.implicit_plane.widget.tubing = tubing
        
        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = True
        module_manager.scalar_lut_manager.show_legend = True

        #scene = self.engine.scenes[0]
        #if ax == 'x_axes': scene.scene.x_plus_view()
        #elif ax == 'y_axes': scene.scene.y_plus_view()
        #else: scene.scene.z_plus_view()

    def surface(self, representation='wireframe', scalar=False, contors = False, n=10):
        mlab.clf()
        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.pipeline.surface(self.src)

        surface = self.engine.scenes[0].children[0].children[0].children[0]
        surface.actor.property.representation = representation
        surface.actor.mapper.scalar_visibility = scalar

        surface.enable_contours = contors
        surface.contour.number_of_contours = n

        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = scalar
        module_manager.scalar_lut_manager.show_legend = scalar

    def contour_plane(self, axis='z', plane=0, contor=100):

        self.axis=axis
        self.plane=plane
        self.contor=contor

        mlab.clf()
        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.pipeline.contour_grid_plane(self.src)

        contour_grid_plane = self.engine.scenes[0].children[0].children[0].children[0]
        contour_grid_plane.actor.mapper.scalar_mode = 'default'
        contour_grid_plane.contour.filled_contours = True

        contour_grid_plane.grid_plane.axis = self.axis
        contour_grid_plane.grid_plane.position = self.plane
        contour_grid_plane.contour.number_of_contours = self.contor

        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = True
        module_manager.scalar_lut_manager.show_legend = True

        scene = self.engine.scenes[0]
        if axis == 'x': scene.scene.x_plus_view()
        elif axis == 'y': scene.scene.y_plus_view()
        else: scene.scene.z_plus_view()

    def streamline(self, u, v, w):
        
        self.u = u
        self.v = v
        self.w = w

        mlab.clf()
        mlab.flow(self.u, self.v, self.w, seedtype='plane')
        streamline = self.engine.scenes[0].children[0].children[0].children[0].children[0]
        streamline.stream_tracer.integration_direction = 'both'
        streamline.stream_tracer.integrator_type = 'runge_kutta45'

        module_manager = self.engine.scenes[0].children[0].children[0].children[0]
        module_manager.vector_lut_manager.show_scalar_bar = True
        module_manager.vector_lut_manager.show_legend = True

    def quiver(self, u, v, w):
        
        self.u = u
        self.v = v
        self.w = w

        mlab.clf()
        mlab.quiver3d(self.x, self.y, self.z, self.u, self.v, self.w)
        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.vector_lut_manager.show_scalar_bar = True
        module_manager.vector_lut_manager.show_legend = True

    def vector_plane(self, u, v, w, masked_points=1, scale=2.0, width=2.0):
        
        self.masked_points = masked_points
        self.scale = scale
        self.width = width

        self.u = u #self.sol[:,:,:,1]
        self.v = v #self.sol[:,:,:,2]
        self.w = w #self.sol[:,:,:,3]

        mlab.clf()
        self.scr = mlab.pipeline.vector_field(self.u, self.v, self.w)
        mlab.pipeline.vector_cut_plane(self.scr, mask_points=self.masked_points, scale_factor=self.scale, line_width=self.width, plane_orientation='y_axes')

        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.vector_lut_manager.show_scalar_bar = True
        module_manager.vector_lut_manager.show_legend = True

    def add_source(self, j):

        self.x, self.y, self.z = [], [], []

        for n in range(self.nb):
            self.x.append(np.array(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], 0, n], dtype='>f4', order='C'))
            self.y.append(np.array(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], 1, n], dtype='>f4', order='C'))
            self.z.append(np.array(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], 2, n], dtype='>f4', order='C'))  

        self.sgrd = [tvtk.StructuredGrid(dimensions=(self.x[i].shape[0], self.x[i].shape[1], self.x[i].shape[2])) for i in range(self.nb)]
        self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
                                    self.y[j].T.ravel(),
                                    self.z[j].T.ravel()] 

        self.src = mlab.pipeline.add_dataset(self.sgrd[j])

    def outline_mb(self, j):
        #mlab.clf()
        self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
                                    self.y[j].T.ravel(),
                                    self.z[j].T.ravel()] 
        self.src = mlab.pipeline.add_dataset(self.sgrd[j])
        mlab.pipeline.outline(self.src)

    def add_parameter_mb(self, j, quantity, name):

        self.quantity = quantity
        self.name = name

        self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
                                    self.y[j].T.ravel(),
                                    self.z[j].T.ravel()] 

        self.sgrd[j].point_data.scalars = self.quantity[:self.ni[j], :self.nj[j], :self.nk[j]].T.ravel()
        self.sgrd[j].point_data.scalars.name = self.name
        self.src = mlab.pipeline.add_dataset(self.sgrd[j])

    def surface_mb(self, j, representation='wireframe', scalar=False):

        self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
                                    self.y[j].T.ravel(),
                                    self.z[j].T.ravel()] 
        self.src = mlab.pipeline.add_dataset(self.sgrd[j])
        mlab.pipeline.surface(self.src)

        surface = self.engine.scenes[0].children[j].children[0].children[0]
        surface.actor.property.representation = representation
        surface.actor.mapper.scalar_visibility = scalar

        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = scalar
        module_manager.scalar_lut_manager.show_legend = scalar

    def quiver_mb(self, j, u, v, w, n=10000):

        self.u = u
        self.v = v
        self.w = w

        self.xx = self.x[j]
        self.yy = self.y[j]
        self.zz = self.z[j]

        mlab.quiver3d(self.xx, self.yy, self.zz, self.u[:self.ni[j], :self.nj[j], :self.nk[j]],
                                                 self.v[:self.ni[j], :self.nj[j], :self.nk[j]],
                                                 self.w[:self.ni[j], :self.nj[j], :self.nk[j]], name='Velocities')

        vectors = self.engine.scenes[0].children[j].children[0].children[0]
        vectors.glyph.mask_input_points = True
        vectors.glyph.mask_points.maximum_number_of_points = n


        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.vector_lut_manager.show_scalar_bar = True
        module_manager.vector_lut_manager.show_legend = True


# class VisualizeMB:
#     
#     def __init__(self, grid):
#         self.grid = grid.grd
#         self.nb = grid.nb
#         self.ni = grid.ni
#         self.nj = grid.nj
#         self.nk = grid.nk
#         
#         #from numpy import array
#         try:
#             self.engine = mayavi.engine
#         except NameError:
#             from mayavi.api import Engine
#             self.engine = Engine()
#             self.engine.start()
#         if len(self.engine.scenes) == 0:
#             self.engine.new_scene()
#         
#         scene = self.engine.scenes[0]
#         scene.scene.show_axes = True
#         
#     def add_source(self, j):
#     
#         self.x, self.y, self.z = [], [], []
#         
#         for n in range(self.nb):
#             self.x.append(np.array(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], 0, n], dtype='>f4', order='C'))
#             self.y.append(np.array(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], 1, n], dtype='>f4', order='C'))
#             self.z.append(np.array(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], 2, n], dtype='>f4', order='C'))  
#         
#         self.sgrd = [tvtk.StructuredGrid(dimensions=(self.x[i].shape[0], self.x[i].shape[1], self.x[i].shape[2])) for i in range(self.nb)]
#         self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
#                                     self.y[j].T.ravel(),
#                                     self.z[j].T.ravel()] 
#         
#         self.src = mlab.pipeline.add_dataset(self.sgrd[j])
#         
#     def view(self, plane):
#         #self.plane = plane
#         scene = self.engine.scenes[0]
#         if plane == 'z+':scene.scene.z_plus_view()
#         elif plane == 'z-':scene.scene.z_minus_view()
#         elif plane== 'x+':scene.scene.x_plus_view()
#         elif plane == 'x-':scene.scene.x_minus_view()
#         elif plane == 'y+':scene.scene.y_plus_view()
#         else: scene.scene.y_minus_view()
#         
#     def outline(self, j):
#         #mlab.clf()
#         self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
#                                     self.y[j].T.ravel(),
#                                     self.z[j].T.ravel()] 
#         self.src = mlab.pipeline.add_dataset(self.sgrd[j])
#         mlab.pipeline.outline(self.src)
#             
#     def add_parameter(self, j, quantity, name):
#     
#         self.quantity = quantity
#         self.name = name
#         
#         self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
#                                     self.y[j].T.ravel(),
#                                     self.z[j].T.ravel()] 
# 
#         self.sgrd[j].point_data.scalars = self.quantity[:self.ni[j], :self.nj[j], :self.nk[j]].T.ravel()
#         self.sgrd[j].point_data.scalars.name = self.name
#         self.src = mlab.pipeline.add_dataset(self.sgrd[j])
#         
#     def surface(self, j, representation='wireframe', scalar=False):
#         
#         self.sgrd[j].points = np.c_[self.x[j].T.ravel(),
#                                     self.y[j].T.ravel(),
#                                     self.z[j].T.ravel()] 
#         self.src = mlab.pipeline.add_dataset(self.sgrd[j])
#         mlab.pipeline.surface(self.src)
#         
#         surface = self.engine.scenes[0].children[j].children[0].children[0]
#         surface.actor.property.representation = representation
#         surface.actor.mapper.scalar_visibility = scalar
#         
#         module_manager = self.engine.scenes[0].children[0].children[0]
#         module_manager.scalar_lut_manager.show_scalar_bar = scalar
#         module_manager.scalar_lut_manager.show_legend = scalar
#         
#     def quiver(self, j, u, v, w, n=10000):
#         
#         self.u = u
#         self.v = v
#         self.w = w
#         
#         self.xx = self.x[j]
#         self.yy = self.y[j]
#         self.zz = self.z[j]
#         
#         mlab.quiver3d(self.xx, self.yy, self.zz, self.u[:self.ni[j], :self.nj[j], :self.nk[j]],
#                                                  self.v[:self.ni[j], :self.nj[j], :self.nk[j]],
#                                                  self.w[:self.ni[j], :self.nj[j], :self.nk[j]], name='Velocities')
#         
#         vectors = self.engine.scenes[0].children[j].children[0].children[0]
#         vectors.glyph.mask_input_points = True
#         vectors.glyph.mask_points.maximum_number_of_points = n
# 
# 
#         module_manager = self.engine.scenes[0].children[0].children[0]
#         module_manager.vector_lut_manager.show_scalar_bar = True
#         module_manager.vector_lut_manager.show_legend = True
#         
# 
