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
    """
    def initUI(self):
        # Widgets

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=np.complex64, obligation=gpi.REQUIRED)
        self.addInPort('coords', 'NPYarray', dtype=np.float32, obligation=gpi.REQUIRED)
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)

    def compute(self):

        import numpy as np
        import bni.gridding.grid_kaiser as dg

        crds = self.getData('coords')
        gdata = self.getData('data')
         
        # assume the last dims (i.e. each image) must be degridded independently
        gdata_iter, iter_shape = self.pinch(gdata, stop=-2)

        # construct an output array w/ slice dims
        out_shape = iter_shape + list(crds.shape)[:-1]
        out = np.zeros(out_shape, dtype=gdata.dtype)
        out_iter,_ = self.pinch(out, stop=-2)

        # degrid all slices
        for i in range(np.prod(iter_shape)):
            out_iter[i] = dg.degrid(crds, gdata_iter[i])

        # reset array shapes and output
        self.setData('out', out)

        return(0)

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
