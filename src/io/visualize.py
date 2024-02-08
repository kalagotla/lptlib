#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from mayavi import mlab
from tvtk.api import tvtk


# In[2]:


class Visualize:
    
    def __init__(self, grd):
        self.grd = grd
        #self.sol = q
        
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
        
    def add_parameter(self, quantity, name):
        self.quantity = quantity
        self.name = name
        mlab.clf()
        #add = self.src.add_attribute(self.quantity.T.ravel(), self.name)
        self.sgrd.point_data.scalars = self.quantity.T.ravel()
        self.sgrd.point_data.scalars.name = self.name
        self.src = mlab.pipeline.add_dataset(self.sgrd)
    
    def grid_extraction(self, xmin, xmax, ymin, ymax, zmin, zmax, representation1='wireframe', scalar1=False):

        mlab.clf()
        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.axes()
        mlab.pipeline.outline(self.src)

        mlab.pipeline.surface(self.src)
        surface = self.engine.scenes[0].children[0].children[0].children[2]
        surface.actor.property.representation = 'points'
        surface.actor.mapper.scalar_visibility = scalar1
        surface.actor.property.opacity = 0.5

        module_manager = self.engine.scenes[0].children[0].children[0]
        module_manager.scalar_lut_manager.show_scalar_bar = scalar1
        module_manager.scalar_lut_manager.show_legend = scalar1

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

        #from mayavi.filters.extract_grid import ExtractGrid
        #extract_grid = ExtractGrid()
        #vtk_data_source = self.engine.scenes[0].children[0]
        #self.engine.add_filter(extract_grid, vtk_data_source)

        mlab.pipeline.extract_grid(self.src)
        extract_grid = self.engine.scenes[0].children[0].children[1]
        extract_grid.x_min = self.xmin
        extract_grid.x_max = self.xmax
        extract_grid.y_min = self.ymin
        extract_grid.y_max = self.ymax
        extract_grid.z_min = self.zmin
        extract_grid.z_max = self.zmax

        mlab.pipeline.surface(extract_grid)
        surface1 = self.engine.scenes[0].children[0].children[1].children[0].children[0]
        surface1.actor.property.representation = representation1
        surface1.actor.mapper.scalar_visibility = scalar1
        surface1.actor.property.color = (0.0, 0.0, 0.0)

    def split(self, axis, position):
        mlab.clf()
        self.axis = axis
        self.position = position

        self.src = mlab.pipeline.add_dataset(self.sgrd)
        mlab.axes()
        mlab.pipeline.outline(self.src)

        mlab.pipeline.grid_plane(self.src)
        grid_plane = self.engine.scenes[0].children[0].children[0].children[2]
        if self.axis == 'i':grid_plane.grid_plane.axis = 'x'
        elif self.axis == 'j':grid_plane.grid_plane.axis = 'y'
        else: grid_plane.grid_plane.axis = 'z'
        grid_plane.grid_plane.position = self.position

        mlab.pipeline.grid_plane(self.src)
        grid_plane = self.engine.scenes[0].children[0].children[0].children[3]
        if self.axis == 'i':grid_plane.grid_plane.axis = 'x'
        elif self.axis == 'j':grid_plane.grid_plane.axis = 'y'
        else: grid_plane.grid_plane.axis = 'z'
        grid_plane.grid_plane.position = 0

        mlab.pipeline.surface(self.src)
        surface = self.engine.scenes[0].children[0].children[0].children[4]
        surface.actor.property.representation = 'points'
        surface.actor.property.opacity = 0.1


# In[ ]:




