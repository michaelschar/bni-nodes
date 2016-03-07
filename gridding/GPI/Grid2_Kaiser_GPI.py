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

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=np.complex64, obligation=gpi.REQUIRED)
        self.addInPort('coords', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        self.addInPort('weights', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)
        self.addOutPort('deapodization', 'NPYarray')

    def validate(self):

        # adjust dims per set
        data = self.getData('data')
        crds = self.getData('coords')
        self.setAttr('dims per set', max=data.ndim)

        return 0

    def compute(self):

        import numpy as np
        import bni.gridding.grid_kaiser as gd

        crds = self.getData('coords').astype(np.float32, copy=False)
        data = self.getData('data').astype(np.complex64, copy=False)
        weights = self.getData('weights').astype(np.float32, copy=False)
        mtx_original = self.getVal('mtx size (n x n)')
        dimsperset = self.getVal('dims per set')
        oversampling_ratio = self.getVal('oversampling ratio')
        fft_and_rolloff = self.getVal('Add FFT and rolloff')

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

        # construct an output array w/ slice dims
        data_iter, iter_shape = self.pinch(data, stop=-dimsperset)
        if iter_shape == []:
            iter_shape = [1]
        
        if fft_and_rolloff:
            # assume the last dims (i.e. each image) must be gridded independently
            # shape = [..., n, n], where 'n' is the image dimensions
            image_shape = [mtx_original, mtx_original]
            out_shape = iter_shape + image_shape
            out = np.zeros(out_shape, dtype=data.dtype)
            out_iter,_ = self.pinch(out, stop=-dimsperset)
            
            # tell the grid routine what shape to produce
            outdim = np.array([mtx,mtx], dtype=np.int64)
            
            # grid all slices
            dx = dy = 0.
            for i in range(int(np.prod(iter_shape))):
                gridded_kspace = gd.grid(crds, data_iter[i], weights, kernel, outdim, dx, dy)
                gridded_kspace = self.fft2(gridded_kspace, dir=0)
                gridded_kspace *=roll
                out_iter[i] = gridded_kspace[mtx_min:mtx_max,mtx_min:mtx_max]
        else:
            # assume the last dims (i.e. each image) must be gridded independently
            # shape = [..., n, n], where 'n' is the image dimensions
            image_shape = [mtx, mtx]
            out_shape = iter_shape + image_shape
            out = np.zeros(out_shape, dtype=data.dtype)
            out_iter,_ = self.pinch(out, stop=-dimsperset)

            # tell the grid routine what shape to produce
            outdim = np.array(image_shape, dtype=np.int64)
            
            # grid all slices
            dx = dy = 0.
            for i in range(int(np.prod(iter_shape))):
                out_iter[i] = gd.grid(crds, data_iter[i], weights, kernel, outdim, dx, dy)
  
        self.setData('out', out.squeeze())

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
