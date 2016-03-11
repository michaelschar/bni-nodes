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
    """DeGridding module for Post-Cartesian Data - works with 2D data.
        First, data are rolloff corrected with the gridding kernel function.
        Second, zeroes are added around the image matrix according to the oversampling factor.
        Third, the data are inverse Fourer transformed and then gridded to the coordinate locations using a Kaiser-Bessel kernel.
        
    WIDGET:
        oversampling ratio: Oversampling and Kaiser-Bessel kernel function according to 
            Beatty, Philip J., Dwight G. Nishimura, and John M. Pauly. "Rapid gridding
            reconstruction with a minimal oversampling ratio." Medical Imaging, IEEE
            Transactions on 24.6 (2005): 799-808.

    INPUT:
        data: data in image domain
        coords: nD array sample locations (scaled between -0.5 and 0.5)
    
    OUTPUT:
        out: k-space resampled at coordinate locations
    """
    def initUI(self):
        # Widgets
        self.addWidget('DoubleSpinBox', 'oversampling ratio', val=1.375, decimals=3, singlestep=0.125, min=1, max=2, collapsed=True)
        
        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=np.complex64, obligation=gpi.REQUIRED)
        self.addInPort('coords', 'NPYarray', dtype=[np.float64, np.float32], obligation=gpi.REQUIRED)
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)

    def compute(self):

        import numpy as np
        import bni.gridding.Kaiser2D_utils as kaiser2D

        # get port and widget inputs
        coords = self.getData('coords').astype(np.float32, copy=False)
        data = self.getData('data').astype(np.complex64, copy=False)
        oversampling_ratio = self.getVal('oversampling ratio')
        
        # Determine matrix size before and after oversampling
        mtx_original = data.shape[-1]
        mtx = np.int(mtx_original * oversampling_ratio)
        if mtx%2:
            mtx+=1
        if oversampling_ratio > 1:
            mtx_min = np.int((mtx-mtx_original)/2)
            mtx_max = mtx_min + mtx_original
        else:
            mtx_min = 0
            mtx_max = mtx
        
        # data dimensions
        nr_points = coords.shape[-2]
        nr_arms = coords.shape[-3]
        if data.ndim == 2:
            nr_coils = 1
            extra_dim1 = 1
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,mtx_original,mtx_original]
        elif data.ndim == 3:
            nr_coils = data.shape[0]
            extra_dim1 = 1
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,mtx_original,mtx_original]
        elif data.ndim == 4:
            nr_coils = data.shape[0]
            extra_dim1 = data.shape[-3]
            extra_dim2 = 1
            data.shape = [nr_coils,extra_dim2,extra_dim1,mtx_original,mtx_original]
        elif data.ndim == 5:
            nr_coils = data.shape[0]
            extra_dim1 = data.shape[-3]
            extra_dim2 = data.shape[-4]
        elif data.ndim > 5:
            self.log.warn("Not implemented yet")
        out_dims_degrid = [nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points]
        out_dims_fft = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]

        # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
        if coords.ndim == 3:
            coords.shape = [1,nr_arms,nr_points,2]

        # pre-calculate Kaiser-Bessel kernel
        kernel_table_size = 800
        kernel = kaiser2D.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)
        
        # pre-calculate the rolloff for the spatial domain
        roll = kaiser2D.rolloff2D(mtx, kernel)

        # perform rolloff correction
        rolloff_corrected_data = data * roll[mtx_min:mtx_max,mtx_min:mtx_max]
    
        # inverse-FFT with zero-interpolation to oversampled k-space
        oversampled_kspace = kaiser2D.fft2D(rolloff_corrected_data, dir=1, out_dims_fft=out_dims_fft)
   
        out = kaiser2D.degrid2D(oversampled_kspace, coords, kernel, out_dims_degrid)
        self.setData('out', out.squeeze())
 
        return(0)

    def execType(self):
        return gpi.GPI_PROCESS