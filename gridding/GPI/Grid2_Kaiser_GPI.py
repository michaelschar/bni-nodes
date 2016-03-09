# Copyright (c) 2014, Dignity Health
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: Nick Zwart
# Date: 2015nov25

import gpi
import numpy as np

class ExternalNode(gpi.NodeAPI):
    """Gridding module for Post-Cartesian Data - works with 2D data.

    WIDGET:
        mtx size (n x n): grid matrix size 'n'
        dims per set: dimensions of the sample coordinates (automatically set)

    INPUT:
        data: nD array of sampled k-space data
        coords: nD array sample locations (scaled between -0.5 and 0.5)
        weights: density compensation
    
    OUTPUT:
        out: gridded k-space
        deapodization: grid kernel compensation to be multiplied by gridded
                       data after fft (if desired).
    """
    def initUI(self):
        # Widgets
        self.addWidget('SpinBox','mtx size (n x n)', min=5, val=240)
        self.addWidget('Slider','dims per set', min=1, val=2)
        self.addWidget('DoubleSpinBox', 'oversampling ratio', val=1.375, decimals=3, singlestep=0.125, min=1, max=2, collapsed=True)
        self.addWidget('PushButton', 'Add FFT and rolloff', toggle=True, button_title='ON', val=1)
        self.addWidget('PushButton', 'New', toggle=True, button_title='ON', val=1)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64,np.complex128], obligation=gpi.REQUIRED)
        self.addInPort('coords', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        self.addInPort('weights', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)
        self.addOutPort('deapodization', 'NPYarray')

    def validate(self):

        # adjust dims per set
        data = self.getData('data')
        coords = self.getData('coords')
        self.setAttr('dims per set', max=data.ndim)
        
        # check size of data vs. coords
        if coords.shape[-1] != 2:
            self.log.warn("Currently only for 2D data")
            return 1
        if coords.shape[-2] != data.shape[-1]:
            self.log.warn("data and coords do not agree in the number of sampled points per arm")
            return 1
        if coords.shape[-3] != data.shape[-2]:
            self.log.warn("data and coords do not agree in the number of arms")
            return 1
            if coords.ndim == 4:
                if data.ndim < 4:
                    self.log.warn("if coords has 4 dimensions then data also needs 4 or more dimensions")
                    return 1
                else:
                    if coords.shape[-4] != data.shape[-3]:
                        self.log.warn("data and coords do not agree in the number of phases / dynamics")
                        return 1

        return 0

    def compute(self):

        import numpy as np
        import bni.gridding.grid_kaiser as gd

        # get port and widget inputs
        coords = self.getData('coords').astype(np.float32, copy=False)
        data = self.getData('data').astype(np.complex64, copy=False)
        weights = self.getData('weights').astype(np.float32, copy=False)
        mtx_original = self.getVal('mtx size (n x n)')
        dimsperset = self.getVal('dims per set')
        oversampling_ratio = self.getVal('oversampling ratio')
        fft_and_rolloff = self.getVal('Add FFT and rolloff')
        new = self.getVal('New')

        # Determine matrix size after oversampling
        mtx = np.int(mtx_original * oversampling_ratio)
        if mtx%2:
            mtx+=1
        if fft_and_rolloff:
            if oversampling_ratio > 1:
                mtx_min = np.int((mtx-mtx_original)/2)
                mtx_max = mtx_min + mtx_original
            else:
                mtx_min = 0
                mtx_max = mtx

        # pre-calculate Kaiser-Bessel kernel
        kernel_table_size = 800
        kernel = self.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)
        # pre-calculate the rolloff for the spatial domain
        roll = self.rolloff2(mtx, kernel)
        self.setData('deapodization', roll)
        
        if new:
            # data dimensions
            nr_points = data.shape[-1]
            nr_arms = data.shape[-2]
            nr_coils = data.shape[0]
            if data.ndim == 3:
                extra_dim1 = 1
                extra_dim2 = 1
                data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
            elif data.ndim == 4:
                extra_dim1 = data.shape[-3]
                extra_dim2 = 1
                data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
            elif data.ndim == 5:
                extra_dim1 = data.shape[-3]
                extra_dim2 = data.shape[-4]
            elif data.ndim > 5:
                self.log.warn("Not implemented yet")
            out_dims = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]
    
            # grid
            gridded_kspace = self.grid2D(data, coords, weights, kernel, out_dims)
            
            if fft_and_rolloff:
                # FFT
                dir = 0
                image_domain = self.fft2D(gridded_kspace, dir, out_dims)
                
                # rolloff
                image_domain *= roll
                self.setData('out', image_domain[...,mtx_min:mtx_max,mtx_min:mtx_max].squeeze())
            
            else:
                self.setData('out', gridded_kspace.squeeze())
            
            
            
        else:

            # construct an output array w/ slice dims
            data_iter, iter_shape = self.pinch(data, stop=-dimsperset)
            if iter_shape == []:
                iter_shape = [1]
            
            # assume the last dims (i.e. each image) must be gridded independently
            # shape = [..., n, n], where 'n' is the image dimensions
            image_shape = [mtx, mtx]
            out_shape = iter_shape + image_shape
            gridded_kspace = np.zeros(out_shape, dtype=data.dtype)
            gridded_kspace_iter,_ = self.pinch(gridded_kspace, stop=-dimsperset)

            # tell the grid routine what shape to produce
            outdim = np.array(image_shape, dtype=np.int64)
            
            # grid all slices
            dx = dy = 0.
            for i in range(int(np.prod(iter_shape))):
                gridded_kspace_iter[i] = gd.grid(coords, data_iter[i], weights, kernel, outdim, dx, dy)

            if fft_and_rolloff:
                # assume the last dims (i.e. each image) must be gridded independently
                # shape = [..., n, n], where 'n' is the image dimensions
                image_shape = [mtx_original, mtx_original]
                out_shape = iter_shape + image_shape
                out = np.zeros(out_shape, dtype=data.dtype)
                out_iter,_ = self.pinch(out, stop=-dimsperset)
                
                for i in range(int(np.prod(iter_shape))):
                    image_domain = self.fft2(gridded_kspace_iter[i], dir=0)
                    image_domain *=roll
                    out_iter[i] = image_domain[mtx_min:mtx_max,mtx_min:mtx_max]
                
                self.setData('out', out.squeeze())
            else:
                self.setData('out', gridded_kspace.squeeze())

        return 0 

    def fft2(self, data, dir=0, zp=1, out_shape=[], tx_ON=True):
        # data: np.complex64
        # dir: int (0 or 1)
        # zp: float (>1)

        # simplify the fftw wrapper
        import numpy as np
        import core.math.fft as corefft

        # generate output dim size array
        # fortran dimension ordering
        outdims = list(data.shape)
        if len(out_shape):
            outdims = out_shape
        else:
            for i in range(len(outdims)):
                outdims[i] = int(outdims[i]*zp)
        outdims.reverse()
        outdims = np.array(outdims, dtype=np.int64)

        # load fft arguments
        kwargs = {}
        kwargs['dir'] = dir

        # transform or just zeropad
        if tx_ON:
            kwargs['dim1'] = 1
            kwargs['dim2'] = 1
        else:
            kwargs['dim1'] = 0
            kwargs['dim2'] = 0

        return corefft.fftw(data, outdims, **kwargs)

    def rolloff2(self, mtx_xy, kernel, clamp_min_percent=5):
        # mtx_xy: int
        import numpy as np
        import bni.gridding.grid_kaiser as gd

        # grid one point at k_0
        dx = dy = 0.0
        coords = np.array([0,0], dtype='float32')
        data = np.array([1.0], dtype='complex64')
        weights = np.array([1.0], dtype='float32')
        outdim = np.array([mtx_xy, mtx_xy],dtype=np.int64)

        # grid -> fft -> |x|
        out = np.abs(self.fft2(gd.grid(coords, data, weights, kernel, outdim, dx, dy)))

        # clamp the lowest values to a percentage of the max
        clamp = out.max() * clamp_min_percent/100.0
        out[out < clamp] = clamp

        # invert
        return 1.0/out

    def execType(self):
        return gpi.GPI_PROCESS

    def pinch(self, a, start=0, stop=-1):
        '''Combine multiple adjacent dimensions into one by taking the product
        of dimension lengths.  The output array is a view of the input array.
        INPUT:
            a: input array
            start: first dimension to pinch
            stop: last dimension to pinch
        OUTPUT:
            out: a view of the input array with pinched dimensions
            iter_shape: a list of dimensions that will be iterated on
        '''
        import numpy as np
        out = a.view()
        s = list(a.shape)
        iter_shape = s[start:stop]
        out_shape = s[:start] + [np.prod(iter_shape)] + s[stop:]
        out.shape = out_shape
        return out, iter_shape

    def kaiserbessel_kernel(self, kernel_table_size, oversampling_ratio):
        #   Generate a Kaiser-Bessel kernel function
        #   OUTPUT: 1D kernel table for radius squared
    
        import bni.gridding.grid_kaiser as dg
        kernel_dim = np.array([kernel_table_size],dtype=np.int64)
        return dg.kaiserbessel_kernel(kernel_dim, np.float64(oversampling_ratio))

    def grid2D(self, data, coords, weights, kernel, out_dims):
        # data: np.float32
        # coords: np.complex64
        # weights: np.float32
        # kernel: np.float64
        # outdims = [nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy]: int
        import bni.gridding.grid_kaiser as bni_grid
        
        [nr_coils, extra_dim2, extra_dim1, mtx_xy, dummy] = out_dims
        
        # off-center in pixels.
        dx = dy = 0.

        # gridded kspace
        gridded_kspace = np.zeros([nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy], dtype=data.dtype)
        
        # tell the grid routine what shape to produce
        outdim = np.array([mtx_xy,mtx_xy], dtype=np.int64)

        # coordinate dimensions
        if coords.ndim == 3:
            same_coords_for_all_slices_and_dynamics = True
            coords.shape = [1,nr_arms,nr_points,2]
        else:
            same_coords_for_all_slices_and_dynamics = False

        # grid all slices
        dx = dy = 0.
        for extra1 in range(extra_dim1):
            if same_coords_for_all_slices_and_dynamics:
                extra1_coords = 0
            else:
                extra1_coords = extra1
            for extra2 in range(extra_dim2):
                for coil in range(nr_coils):
                    gridded_kspace[coil,extra2,extra1,:,:] = bni_grid.grid(coords[extra1_coords,:,:,:], data[coil,extra2,extra1,:,:], weights, kernel, outdim, dx, dy)

        return gridded_kspace

    def fft2D(self, data, dir=0, outdims=[], fft_or_zeropad=True):
        # data: np.complex64
        # dir: int (0 or 1)
        # outdims = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]

        import core.math.fft as corefft

        # generate output dim size array
        # fortran dimension ordering
        outdims.reverse()
        outdims = np.array(outdims, dtype=np.int64)

        # load fft arguments
        kwargs = {}
        kwargs['dir'] = dir

        # transform
        kwargs['dim1'] = 1
        kwargs['dim2'] = 1
        kwargs['dim3'] = 0
        kwargs['dim4'] = 0
        kwargs['dim5'] = 0

        return corefft.fftw(data, outdims, **kwargs)


