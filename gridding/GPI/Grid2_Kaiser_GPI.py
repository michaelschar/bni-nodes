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
        oversampling ratio: Oversampling and Kaiser-Bessel kernel function according to
            Beatty, Philip J., Dwight G. Nishimura, and John M. Pauly. "Rapid gridding
            reconstruction with a minimal oversampling ratio." Medical Imaging, IEEE
            Transactions on 24.6 (2005): 799-808.
        Add FFT and rolloff: push button to perform both FFT and rolloff correction and output the cropped image.

    INPUT:
        data: nD array of sampled k-space data
        coords: nD array sample locations (scaled between -0.5 and 0.5)
        weights: density compensation
        
    
    OUTPUT:
        out: gridded k-space or image cropped to demanded matrix size
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
        import bni.gridding.Kaiser2D_utils as kaiser2D

        # get port and widget inputs
        coords = self.getData('coords').astype(np.float32, copy=False)
        data = self.getData('data').astype(np.complex64, copy=False)
        weights = self.getData('weights').astype(np.float32, copy=False)
        mtx_original = self.getVal('mtx size (n x n)')
        dimsperset = self.getVal('dims per set')
        oversampling_ratio = self.getVal('oversampling ratio')
        fft_and_rolloff = self.getVal('Add FFT and rolloff')

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
        kernel = kaiser2D.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)
        # pre-calculate the rolloff for the spatial domain
        roll = kaiser2D.rolloff2D(mtx, kernel)
        self.setData('deapodization', roll)
        
        # data dimensions
        nr_points = data.shape[-1]
        nr_arms = data.shape[-2]
        nr_coils = data.shape[0]
        if data.ndim == 2:
            nr_coils = 1
            extra_dim1 = 1
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
        elif data.ndim == 3:
            nr_coils = data.shape[0]
            extra_dim1 = 1
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
        elif data.ndim == 4:
            nr_coils = data.shape[0]
            extra_dim1 = data.shape[-3]
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,nr_arms,nr_points]
        elif data.ndim == 5:
            nr_coils = data.shape[0]
            extra_dim1 = data.shape[-3]
            extra_dim2 = data.shape[-4]
        elif data.ndim > 5:
            self.log.warn("Not implemented yet")
        out_dims_grid = [nr_coils, extra_dim2, extra_dim1, mtx, nr_arms, nr_points]
        out_dims_fft = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]

        # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
        if coords.ndim == 3:
            coords.shape = [1,nr_arms,nr_points,2]
            weights.shape = [1,nr_arms,nr_points]
        
        # grid
        self.log.debug("before gridding")
        gridded_kspace = kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid)
        self.log.debug("after gridding")
        if fft_and_rolloff:
            # FFT
            image_domain = kaiser2D.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
            self.log.debug("after fft")
            # rolloff
            image_domain *= roll
            self.log.debug("after roll")
            self.setData('out', image_domain[...,mtx_min:mtx_max,mtx_min:mtx_max].squeeze())
        
        else:
            self.setData('out', gridded_kspace.squeeze())
            
        return 0 

    def execType(self):
        return gpi.GPI_PROCESS