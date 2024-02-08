#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
from types import SimpleNamespace
from visualize import *
# In[3]:


class GridExtraction:
    
    def __init__(self, grid, flow):
        self.grid = grid.grd
        self.q = flow.q
        
        self.nb = grid.nb
        self.ni = grid.ni
        self.nj = grid.nj
        self.nk = grid.nk
        
        self.mach = flow.mach
        self.alpha = flow.alpha
        self.rey = flow.rey
        self.time = flow.time
        
        self.gd, self.fl = [], []
            
        self.df = pd.DataFrame(columns=['I_max', 'J_max', 'K_max'])

        for n in range(int(self.nb)):
            self.gd.append(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], :, n])
            self.fl.append(self.q[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], :, n])
            
            self.df.loc[len(self.df)] = ([self.ni[n], self.nj[n], self.nk[n]])
            
        print(self.df)
        
    def block_num(self, num):
        self.num = num
        self.v = Visualize(self.gd[self.num])

    def extract(self, x_data=[None, None], y_data=[None, None], z_data=[None, None],
                quantity=None, name=None,
                representation='wireframe', scalar=False):
        
        self.x_data = x_data
        self.y_data = y_data
        self.z_data = z_data
        
        self.name = name
        self.quantity = quantity
        
        if self.x_data == [None, None]: self.x_data = [0,self.ni[self.num]]
        if self.y_data == [None, None]: self.y_data = [0,self.nj[self.num]]
        if self.z_data == [None, None]: self.z_data = [0,self.nk[self.num]]

        self.ext_grid = self.gd[self.num][self.x_data[0]:self.x_data[1], 
                                          self.y_data[0]:self.y_data[1],
                                          self.z_data[0]:self.z_data[1], :]
        print("Grid data reading is successful! grid.shape: ", self.ext_grid.shape)
        
        self.ext_q = self.fl[self.num][self.x_data[0]:self.x_data[1],
                                       self.y_data[0]:self.y_data[1],
                                       self.z_data[0]:self.z_data[1], :]
        print("\nFlow file read successfully! flow.shape: ",self.ext_q.shape)
        
        if self.name != None: self.v.add_parameter(self.quantity, self.name)
        self.v.grid_extraction(self.x_data[0], self.x_data[1]-1,
                          self.y_data[0], self.y_data[1]-1,
                          self.z_data[0], self.z_data[1]-1,
                          representation, scalar)
        
    def create_grid(self):
        self.grid = SimpleNamespace()
        self.grid.nb = 1
        self.grid.ni = self.ext_grid.shape[0]
        self.grid.nj = self.ext_grid.shape[1]
        self.grid.nk = self.ext_grid.shape[2]
        self.grid.grd = self.ext_grid
        return self.grid
    
    def create_flow(self):
        self.flow = SimpleNamespace()
        self.flow.nb = 1
        self.flow.ni = self.ext_q.shape[0]
        self.flow.nj = self.ext_q.shape[1]
        self.flow.nk = self.ext_q.shape[2]
        self.flow.mach = self.mach
        self.flow.alpha = self.alpha
        self.flow.rey = self.rey
        self.flow.time = self.time
        self.flow.q = self.ext_q
        return self.flow 


# In[4]:


class SplitBlock:
    
    def __init__(self, grid, flow):
        self.grid = grid.grd
        self.q = flow.q
        
        self.nb = grid.nb
        self.ni = grid.ni
        self.nj = grid.nj
        self.nk = grid.nk
        
        self.mach = flow.mach
        self.alpha = flow.alpha
        self.rey = flow.rey
        self.time = flow.time
        
        self.gd, self.fl = [], []
            
        self.df = pd.DataFrame(columns=['I_max', 'J_max', 'K_max'])

        for n in range(int(self.nb)):
            self.gd.append(self.grid[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], :, n])
            self.fl.append(self.q[0:self.ni[n], 0:self.nj[n], 0:self.nk[n], :, n])
            
            self.df.loc[len(self.df)] = ([self.ni[n]-1, self.nj[n]-1, self.nk[n]-1])
            
        print(self.df)
        
    def block_num(self, num):
        self.num = num
        self.v = Visualize(self.gd[self.num])
    
    def split(self, direct, plane):
        
        self.direct = direct
        self.plane = plane
        
        self.v.split(self.direct, self.plane)
        
        self.new_grid, self.new_q = [], []
        for n in range(int(self.nb+1)):
            self.new_grid.append([])
            self.new_q.append([])
            
        self.new_ni = np.zeros(self.nb+1).astype(int)
        self.new_nj = np.zeros(self.nb+1).astype(int)
        self.new_nk = np.zeros(self.nb+1).astype(int)
        
        if self.direct == 'i':
            #self.plane = self.find_nearest_x(self.x[self.num], self.plane, self.num)
            self.block_1 = self.gd[self.num][0:self.plane, :, :, :]
            self.block_2 = self.gd[self.num][self.plane-1:, :, :, :]
            
            self.block_1_q = self.fl[self.num][0:self.plane, :, :, :]
            self.block_2_q = self.fl[self.num][self.plane-1:, :, :, :]
            
            if self.num == 0:
                self.new_grid[0] = self.block_1
                self.new_grid[1] = self.block_2
                self.new_grid[2:] = self.gd[self.num+1:]
                
                self.new_q[0] = self.block_1_q
                self.new_q[1] = self.block_2_q
                self.new_q[2:] = self.fl[self.num+1:]
                
                self.new_ni[0] = self.block_1.shape[0]
                self.new_ni[1] = self.block_2.shape[0]
                self.new_ni[2:] = self.ni[self.num+1:]
                
                self.new_nj[0] = self.block_1.shape[1]
                self.new_nj[1] = self.block_2.shape[1]
                self.new_nj[2:] = self.nj[self.num+1:]
                
                self.new_nk[0] = self.block_1.shape[2]
                self.new_nk[1] = self.block_2.shape[2]
                self.new_nk[2:] = self.nk[self.num+1:]
                
            if self.num == int(self.nb-1):
                self.new_grid[0:self.num] = self.gd[0:self.num]
                self.new_grid[self.num] = self.block_1
                self.new_grid[self.num+1] = self.block_2
                
                self.new_q[0:self.num] = self.fl[0:self.num]
                self.new_q[self.num] = self.block_1_q
                self.new_q[self.num+1] = self.block_2_q
                
                self.new_ni[0:self.num] = self.ni[0:self.num]
                self.new_ni[self.num] = self.block_1.shape[0]
                self.new_ni[self.num+1] = self.block_2.shape[0]
                
                self.new_nj[0:self.num] = self.nj[0:self.num]
                self.new_nj[self.num] = self.block_1.shape[1]
                self.new_nj[self.num+1] = self.block_2.shape[1]
                
                self.new_nk[0:self.num] = self.nk[0:self.num]
                self.new_nk[self.num] = self.block_1.shape[2]
                self.new_nk[self.num+1] = self.block_2.shape[2]
                
            else:
                self.new_grid[0:self.num] = self.gd[0:self.num]
                self.new_grid[self.num] = self.block_1
                self.new_grid[self.num+1] = self.block_2
                self.new_grid[self.num+2:] = self.gd[self.num+1:]
                
                self.new_q[0:self.num] = self.fl[0:self.num]
                self.new_q[self.num] = self.block_1_q
                self.new_q[self.num+1] = self.block_2_q
                self.new_q[self.num+2:] = self.fl[self.num+1:]
                
                self.new_ni[0:self.num] = self.ni[0:self.num]
                self.new_ni[self.num] = self.block_1.shape[0]
                self.new_ni[self.num+1] = self.block_2.shape[0]
                self.new_ni[self.num+2:] = self.ni[self.num+1:]
                
                self.new_nj[0:self.num] = self.nj[0:self.num]
                self.new_nj[self.num] = self.block_1.shape[1]
                self.new_nj[self.num+1] = self.block_2.shape[1]
                self.new_nj[self.num+2:] = self.nj[self.num+1:]
                
                self.new_nk[0:self.num] = self.nk[0:self.num]
                self.new_nk[self.num] = self.block_1.shape[2]
                self.new_nk[self.num+1] = self.block_2.shape[2]
                self.new_nk[self.num+2:] = self.nk[self.num+1:]
                
        if self.direct == 'j':
            #self.plane = self.find_nearest_y(self.y[self.num], self.plane, self.num)
            self.block_1 = self.gd[self.num][:, 0:self.plane, :, :]
            self.block_2 = self.gd[self.num][:, self.plane-1:, :, :]
            
            self.block_1_q = self.fl[self.num][:, 0:self.plane, :, :]
            self.block_2_q = self.fl[self.num][:, self.plane-1:, :, :]
            
            if self.num == 0:
                self.new_grid[0] = self.block_1
                self.new_grid[1] = self.block_2
                self.new_grid[2:] = self.gd[self.num+1:]
                
                self.new_q[0] = self.block_1_q
                self.new_q[1] = self.block_2_q
                self.new_q[2:] = self.fl[self.num+1:]
                
                self.new_ni[0] = self.block_1.shape[0]
                self.new_ni[1] = self.block_2.shape[0]
                self.new_ni[2:] = self.ni[self.num+1:]
                
                self.new_nj[0] = self.block_1.shape[1]
                self.new_nj[1] = self.block_2.shape[1]
                self.new_nj[2:] = self.nj[self.num+1:]
                
                self.new_nk[0] = self.block_1.shape[2]
                self.new_nk[1] = self.block_2.shape[2]
                self.new_nk[2:] = self.nk[self.num+1:]
                
            if self.num == int(self.nb-1):
                self.new_grid[0:self.num] = self.gd[0:self.num]
                self.new_grid[self.num] = self.block_1
                self.new_grid[self.num+1] = self.block_2
                
                self.new_q[0:self.num] = self.fl[0:self.num]
                self.new_q[self.num] = self.block_1_q
                self.new_q[self.num+1] = self.block_2_q
                
                self.new_ni[0:self.num] = self.ni[0:self.num]
                self.new_ni[self.num] = self.block_1.shape[0]
                self.new_ni[self.num+1] = self.block_2.shape[0]
                
                self.new_nj[0:self.num] = self.nj[0:self.num]
                self.new_nj[self.num] = self.block_1.shape[1]
                self.new_nj[self.num+1] = self.block_2.shape[1]
                
                self.new_nk[0:self.num] = self.nk[0:self.num]
                self.new_nk[self.num] = self.block_1.shape[2]
                self.new_nk[self.num+1] = self.block_2.shape[2]
                
            else:
                self.new_grid[0:self.num] = self.gd[0:self.num]
                self.new_grid[self.num] = self.block_1
                self.new_grid[self.num+1] = self.block_2
                self.new_grid[self.num+2:] = self.gd[self.num+1:]
                
                self.new_q[0:self.num] = self.fl[0:self.num]
                self.new_q[self.num] = self.block_1_q
                self.new_q[self.num+1] = self.block_2_q
                self.new_q[self.num+2:] = self.fl[self.num+1:]
                
                self.new_ni[0:self.num] = self.ni[0:self.num]
                self.new_ni[self.num] = self.block_1.shape[0]
                self.new_ni[self.num+1] = self.block_2.shape[0]
                self.new_ni[self.num+2:] = self.ni[self.num+1:]
                
                self.new_nj[0:self.num] = self.nj[0:self.num]
                self.new_nj[self.num] = self.block_1.shape[1]
                self.new_nj[self.num+1] = self.block_2.shape[1]
                self.new_nj[self.num+2:] = self.nj[self.num+1:]
                
                self.new_nk[0:self.num] = self.nk[0:self.num]
                self.new_nk[self.num] = self.block_1.shape[2]
                self.new_nk[self.num+1] = self.block_2.shape[2]
                self.new_nk[self.num+2:] = self.nk[self.num+1:]
                
        if self.direct == 'k':
            #self.plane = self.find_nearest_z(self.z[self.num], self.plane)
            self.block_1 = self.gd[self.num][:, :, 0:self.plane, :]
            self.block_2 = self.gd[self.num][:, :, self.plane-1:, :]
            
            self.block_1_q = self.fl[self.num][:, :, 0:self.plane, :]
            self.block_2_q = self.fl[self.num][:, :, self.plane-1:, :]
            
            if self.num == 0:
                self.new_grid[0] = self.block_1
                self.new_grid[1] = self.block_2
                self.new_grid[2:] = self.gd[self.num+1:]
                
                self.new_q[0] = self.block_1_q
                self.new_q[1] = self.block_2_q
                self.new_q[2:] = self.fl[self.num+1:]
                
                self.new_ni[0] = self.block_1.shape[0]
                self.new_ni[1] = self.block_2.shape[0]
                self.new_ni[2:] = self.ni[self.num+1:]
                
                self.new_nj[0] = self.block_1.shape[1]
                self.new_nj[1] = self.block_2.shape[1]
                self.new_nj[2:] = self.nj[self.num+1:]
                
                self.new_nk[0] = self.block_1.shape[2]
                self.new_nk[1] = self.block_2.shape[2]
                self.new_nk[2:] = self.nk[self.num+1:]
                
            if self.num == int(self.nb-1):
                self.new_grid[0:self.num] = self.gd[0:self.num]
                self.new_grid[self.num] = self.block_1
                self.new_grid[self.num+1] = self.block_2
                
                self.new_q[0:self.num] = self.fl[0:self.num]
                self.new_q[self.num] = self.block_1_q
                self.new_q[self.num+1] = self.block_2_q
                
                self.new_ni[0:self.num] = self.ni[0:self.num]
                self.new_ni[self.num] = self.block_1.shape[0]
                self.new_ni[self.num+1] = self.block_2.shape[0]
                
                self.new_nj[0:self.num] = self.nj[0:self.num]
                self.new_nj[self.num] = self.block_1.shape[1]
                self.new_nj[self.num+1] = self.block_2.shape[1]
                
                self.new_nk[0:self.num] = self.nk[0:self.num]
                self.new_nk[self.num] = self.block_1.shape[2]
                self.new_nk[self.num+1] = self.block_2.shape[2]
                
            else:
                self.new_grid[0:self.num] = self.gd[0:self.num]
                self.new_grid[self.num] = self.block_1
                self.new_grid[self.num+1] = self.block_2
                self.new_grid[self.num+2:] = self.gd[self.num+1:]
                
                self.new_q[0:self.num] = self.fl[0:self.num]
                self.new_q[self.num] = self.block_1_q
                self.new_q[self.num+1] = self.block_2_q
                self.new_q[self.num+2:] = self.fl[self.num+1:]
                
                self.new_ni[0:self.num] = self.ni[0:self.num]
                self.new_ni[self.num] = self.block_1.shape[0]
                self.new_ni[self.num+1] = self.block_2.shape[0]
                self.new_ni[self.num+2:] = self.ni[self.num+1:]
                
                self.new_nj[0:self.num] = self.nj[0:self.num]
                self.new_nj[self.num] = self.block_1.shape[1]
                self.new_nj[self.num+1] = self.block_2.shape[1]
                self.new_nj[self.num+2:] = self.nj[self.num+1:]
                
                self.new_nk[0:self.num] = self.nk[0:self.num]
                self.new_nk[self.num] = self.block_1.shape[2]
                self.new_nk[self.num+1] = self.block_2.shape[2]
                self.new_nk[self.num+2:] = self.nk[self.num+1:]
                
    def create_grid(self):
        self.grid = SimpleNamespace()
        self.grid.nb = self.nb+1
        self.grid.ni = self.new_ni
        self.grid.nj = self.new_nj
        self.grid.nk = self.new_nk
        self.grid.nt = self.grid.ni * self.grid.nj * self.grid.nk * 3

        self.grid.list = []
        for n in range(int(self.grid.nb)):
            self.grid.list.extend(np.ravel(self.new_grid[n], order='F'))
        self.grid._temp = np.array(self.grid.list, dtype='f4')
        
        self.grid.grd = np.zeros((self.grid.ni.max(), self.grid.nj.max(), self.grid.nk.max(), 3, self.grid.nb))
        
        for n in range(int(self.grid.nb)):
            self.grid.grd[0:self.grid.ni[n], 0:self.grid.nj[n], 0:self.grid.nk[n], 0:3, n] =                 self.grid._temp[sum(self.grid.nt[0:n]):sum(self.grid.nt[0:n]) + self.grid.nt[n]]                 .reshape((self.grid.ni[n], self.grid.nj[n], self.grid.nk[n], 3), order='F')

        return self.grid
    
    def create_flow(self):
        self.flow = SimpleNamespace()

        self.flow.nb = self.nb+1
        self.flow.ni = self.new_ni
        self.flow.nj = self.new_nj
        self.flow.nk = self.new_nk

        self.flow.mach = self.mach
        self.flow.alpha = self.alpha
        self.flow.rey = self.rey
        self.flow.time = self.time
        
        self.flow.nt = self.flow.ni * self.flow.nj * self.flow.nk * 5
            
        self.flow.list_q = []
        for n in range(int(self.flow.nb)):
            self.flow.list_q.extend([self.mach, self.alpha, self.rey, self.time])
            self.flow.list_q.extend(np.ravel(self.new_q[n], order='F'))
            
        self.flow._temporary = np.array(self.flow.list_q, dtype='f4')
        
        self.flow.q = np.zeros((self.flow.ni.max(), self.flow.nj.max(), self.flow.nk.max(), 5, self.flow.nb))
        
        for n in range(int(self.flow.nb)):
            self.flow.q[0:self.flow.ni[n], 0:self.flow.nj[n], 0:self.flow.nk[n], 0:5, n] =                 self.flow._temporary[sum(self.flow.nt[0:n])+((n+1)*4):sum(self.flow.nt[0:n]) + self.flow.nt[n] + ((n+1)*4)]                 .reshape((self.flow.ni[n], self.flow.nj[n], self.flow.nk[n], 5), order='F')

        return self.flow 

