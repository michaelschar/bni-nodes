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

import numpy as np
import gpi

class ExternalNode(gpi.NodeAPI):
    """2D iterative SENSE reconstruction.

    * Pruessmann, Klaas P., et al. "Advances in sensitivity encoding with
      arbitrary k‐space trajectories." Magnetic Resonance in Medicine 46.4
      (2001): 638-651.
    * Shewchuk, Jonathan Richard. "An introduction to the conjugate gradient
      method without the agonizing pain." (1994).

    INPUT:
        data - raw k-space data
        crds - trajectory coordinates scaled from -0.5 to 0.5
        weights - sample density weights for gridding
        coil sensitivity - non-conjugated sensitivity maps
    OUTPUT:
        
    WIDGETS:
        mtx - the matrix to be used for gridding (this is the size used no
              extra scaling is added)
    """

    def initUI(self):
        # Widgets
        self.addWidget('SpinBox', 'mtx', val=300, min=1)
        self.addWidget('SpinBox', 'iterations', val=10, min=1)
        self.addWidget('PushButton', 'step')
        self.addWidget('Slider', 'Autocalibration Width (%)', val=3, min=0, max=100)
        self.addWidget('Slider', 'Autocalibration Taper (%)', val=25, min=0, max=100)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=np.complex64)
        self.addInPort('crds', 'NPYarray', dtype=np.float32)
        self.addInPort('weights', 'NPYarray', dtype=np.float32)
        self.addInPort('coil sensitivity', 'NPYarray', dtype=np.complex64, obligation=gpi.OPTIONAL)
        self.addOutPort('x', 'NPYarray', ndim=2, dtype=np.complex64)
        self.addOutPort('r', 'NPYarray', ndim=2, dtype=np.complex64)
        self.addOutPort('d', 'NPYarray', ndim=2, dtype=np.complex64)
        self.addOutPort('Autocalibrated CSM', 'NPYarray', dtype=np.complex64)

    def validate(self):

        iterations = self.getVal('iterations')
        step = self.getVal('step')

        if step and (self.getData('d') is not None):
            # update the UI with the number of iterations
            self.setAttr('iterations', quietval=iterations+1)

        csm = self.getData('coil sensitivity')
        if csm is None:
            self.setAttr('Autocalibration Width (%)', visible=True)
            self.setAttr('Autocalibration Taper (%)', visible=True)
        else:
            self.setAttr('Autocalibration Width (%)', visible=False)
            self.setAttr('Autocalibration Taper (%)', visible=False)

    def compute(self):
        
        print('Start SENSE')
        data = self.getData('data')
        coords = self.getData('crds')
        weights = self.getData('weights')
        csm = self.getData('coil sensitivity')
        mtx = self.getVal('mtx')
        iterations = self.getVal('iterations')
        step = self.getVal('step')

        # pre-calculate the rolloff for the spatial domain
        roll = self.rolloff2(mtx)

        # grid (loop over coils) to create images that are corrupted by
        # aliasing due to undersampling.  If the k-space data have an
        # auto-calibration region, then this can be used to generate B1 maps.
        images = self.loop2(self.grid2, data, mtx, coords, weights)
        images = self.loop2(self.fft2, images, dir=0)
        images *= roll

        # calculate auto-calibration B1 maps
        if csm is None:
            print('\tGenerating autocalibrated B1 maps...')
            csm = self.autocalibrationB1Maps(images)
        else:
            # make sure input csm are the same mtx size
            csm = self.pad2(csm, mtx)
        self.setData('Autocalibrated CSM', csm)

        # keep a conjugate csm set on hand
        csm_conj = np.conj(csm)

        ## Iteration 1:
        if step and (self.getData('d') is not None):
            print('\tSENSE Iteration: ', iterations)
            # make sure the loop doesn't start if only one step is needed
            iterations = 0

            # Get the data from the last execution of this node for an
            # additional single iteration.
            d = self.getData('d').copy()
            r = self.getData('r').copy()
            x = self.getData('x').copy()

            # A
            Ad = csm * d # add coil phase
            Ad *= roll # pre-rolloff for degrid convolution
            Ad = self.loop2(self.fft2, Ad, dir=1)
            Ad = self.loop2(self.degrid2, Ad, coords)
            Ad = self.loop2(self.grid2, Ad, mtx, coords, weights)
            Ad = self.loop2(self.fft2, Ad, dir=0)
            Ad *= roll
            Ad = csm_conj * Ad # broadcast multiply to remove coil phase
            Ad = Ad.sum(axis=0) # assume the coil dim is the first

        else:
            print('\tSENSE Iteration: ', 1)
            # calculate initial conditions
            # d_0
            d_0 = csm_conj * images # broadcast multiply to remove coil phase
            d_0 = d_0.sum(axis=0) # assume the coil dim is the first

            # Ad_0:
            #   degrid -> grid (loop over coils)
            Ad_0 = csm * d_0 # add coil phase
            Ad_0 *= roll # pre-rolloff for degrid convolution
            Ad_0 = self.loop2(self.fft2, Ad_0, dir=1)
            Ad_0 = self.loop2(self.degrid2, Ad_0, coords)
            Ad_0 = self.loop2(self.grid2, Ad_0, mtx, coords, weights)
            Ad_0 = self.loop2(self.fft2, Ad_0, dir=0)
            Ad_0 *= roll
            Ad_0 = csm_conj * Ad_0 # broadcast multiply to remove coil phase
            Ad_0 = Ad_0.sum(axis=0) # assume the coil dim is the first

            # use the initial conditions for the first iter
            r = d = d_0
            x = np.zeros_like(d)
            Ad = Ad_0

        # CG - iter 1
        d_last, r_last, x_last = self.do_cg(d, r, x, Ad)

        ## Iterations >1:
        for i in range(iterations-1):
            print('\tSENSE Iteration: ', i+2)

            # input the result of the last iter
            d = d_last
            r = r_last
            x = x_last

            # A
            Ad = csm * d # add coil phase
            Ad *= roll # pre-rolloff for degrid convolution
            Ad = self.loop2(self.fft2, Ad, dir=1)
            Ad = self.loop2(self.degrid2, Ad, coords)
            Ad = self.loop2(self.grid2, Ad, mtx, coords, weights)
            Ad = self.loop2(self.fft2, Ad, dir=0)
            Ad *= roll
            Ad = csm_conj * Ad # broadcast multiply to remove coil phase
            Ad = Ad.sum(axis=0) # assume the coil dim is the first

            # CG
            d_last, r_last, x_last = self.do_cg(d, r, x, Ad)

        # return the final image     
        self.setData('d', d_last)
        self.setData('r', r_last)
        self.setData('x', x_last)

        return 0

    def do_cg(self, d_in, r_in, x_in, Ad_in):

        d_out = np.zeros_like(d_in)
        r_out = np.zeros_like(r_in)
        x_out = np.zeros_like(x_in)

        # Calculate alpha
        # r^H r / (d^H Ad)
        rHr = np.dot(np.conj(r_in.flatten()), r_in.flatten()) 
        dHAd = np.dot(np.conj(d_in.flatten()), Ad_in.flatten())
        alpha = rHr/dHAd

        # Calculate x(i+1)
        # x(i) + alpha d(i)
        x_out = x_in + alpha * d_in

        # Calculate r(i+1)
        # r(i) - alpha Ad(i)
        r_out = r_in - alpha * Ad_in

        # Calculate beta
        # r(i+1)^H r(i+1) / (r(i)^H r(i))
        r1Hr1 = np.dot(np.conj(r_out.flatten()), r_out.flatten()) 
        beta = r1Hr1 / rHr

        # Calculate d(i+1)
        # r(i+1) + beta d(i)
        d_out = r_out + beta * d_in

        return (d_out, r_out, x_out)

    def loop2(self, func, data, *args, **kwargs):
        # Loop over the extra dimensions of the data, append the output
        # of func to the final output for each of the looped dimensions.
        # 'data' must be the first arg to func.
        # 'func' must return one numpy array
        import numpy as np

        # concatenate all dims to be looped on
        orig_shape = list(data.shape)
        loop_dims = orig_shape[0:-2]
        loops = np.prod(loop_dims)
        loop_shape = [loops] + orig_shape[-2:]
        data.shape = loop_shape

        outdata = None
        for i in range(loops):
            fresult = func(data[i], *args, **kwargs)
            fresult = np.expand_dims(fresult, 0) # add a dim for appending to
            if i == 0:
                outdata = fresult.copy()
            else:
                outdata = np.append(outdata, fresult, axis=0)

        # reset input shape
        data.shape = orig_shape

        # replace loop dims with original dims
        outdata.shape = loop_dims + list(outdata.shape)[-2:]

        return outdata

    def pad2(self, img, mtx):
        out_shape = list(img.shape)
        out_shape[-1] = mtx
        out_shape[-2] = mtx
        return self.fft2(img, out_shape=out_shape, tx_ON=False)

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

    def rolloff2(self, mtx_xy, clamp_min_percent=5):
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
        out = np.abs(self.fft2(gd.grid(coords,data,weights,outdim,dx,dy)))

        # clamp the lowest values to a percentage of the max
        clamp = out.max() * clamp_min_percent/100.0
        out[out < clamp] = clamp

        # invert
        return 1.0/out

    def grid2(self, data, mtx_xy, coords, weights):
        # mtx_xy: int
        # coords: np.float32
        # data: np.complex64
        # weights: np.float32

        # check dims
        assert list(data.shape) == list(coords.shape)[:-1]
        assert list(data.shape) == list(weights.shape)

        import numpy as np
        import bni.gridding.grid_kaiser as gd
        dx = dy = 0.0
        outdim = np.array([mtx_xy, mtx_xy],dtype=np.int64)
        return gd.grid(coords,data,weights,outdim,dx,dy)

    def degrid2(self, gridded_data, coords):
        import bni.gridding.grid_kaiser as dg
        return dg.degrid(coords, gridded_data)

    def execType(self):
        # numpy linalg fails if this isn't a thread :(
        #return gpi.GPI_THREAD
        return gpi.GPI_PROCESS

    def autocalibrationB1Maps(self, images):
        # get UI params
        taper = self.getVal('Autocalibration Taper (%)')
        width = self.getVal('Autocalibration Width (%)')

        # generate window function for blurring image data
        win = self.window2(images.shape[-2:], windowpct=taper, widthpct=width)

        # apply kspace filter
        kspace = self.loop2(self.fft2, images, dir=1)
        kspace *= win

        # transform back into image space and normalize
        csm = self.loop2(self.fft2, kspace, dir=0)
        csm = csm / np.sqrt(np.sum(np.abs(csm)**2, axis=0))

        return csm

    def window2(self, shape, windowpct=100.0, widthpct=100.0, stopVal=0, passVal=1):
        # 2D hanning window just like shapes
        #   OUTPUT: 2D float32 circularly symmetric hanning

        import numpy as np

        # window function width
        bnd = 100.0/widthpct

        # generate coords for each dimension
        x = np.linspace(-bnd, bnd, shape[-1], endpoint=(shape[-1] % 2 != 0))
        y = np.linspace(-bnd, bnd, shape[-2], endpoint=(shape[-2] % 2 != 0))

        # create a 2D grid with coordinates then get radial coords
        xx, yy = np.meshgrid(x,y)
        radius = np.sqrt(xx*xx + yy*yy)

        # calculate hanning
        windIdx = radius <= 1.0
        passIdx = radius <= (1.0 - (windowpct/100.0))
        func = 0.5 * (1.0 - np.cos(np.pi * (1.0 - radius[windIdx]) / (windowpct/100.0)))

        # populate output array
        out = np.zeros(shape, dtype=np.float32)
        out[windIdx] = stopVal + func * (passVal - stopVal)
        out[passIdx] = passVal

        return out
