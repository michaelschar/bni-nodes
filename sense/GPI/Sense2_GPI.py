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

    WIDGETS:
        mtx: the matrix to be used for gridding (this is the size used no
              extra scaling is added)
        iterations: number of iterations to complete before terminating
        step: execute an additional iteration (will add to 'iterations')
        Autocalibration Width (%): percentage of pixels to use for B1 est.
        Autocalibration Taper (%): han window taper for blurring.

    INPUT:
        data: raw k-space data
        coords: trajectory coordinates scaled from -0.5 to 0.5
        weights: sample density weights for gridding
        coil sensitivity: non-conjugated sensitivity maps

    OUTPUT:
        x: solution at the current iteration
        r: residualt at the current iteration
        d: direction at the current iteration
        Autocalibration CSM: B1-recv estimated using the central k-space points
    """

    def initUI(self):
        # Widgets
        self.addWidget('SpinBox', 'mtx', val=300, min=1)
        self.addWidget('SpinBox', 'iterations', val=10, min=1)
        self.addWidget('PushButton', 'step')
        self.addWidget('DoubleSpinBox', 'oversampling ratio', val=1.375, decimals=3, singlestep=0.125, min=1, max=2, collapsed=True)
        self.addWidget('Slider', 'Autocalibration Width (%)', val=10, min=0, max=100)
        self.addWidget('Slider', 'Autocalibration Taper (%)', val=50, min=0, max=100)
        self.addWidget('Slider', 'Mask Floor (% of max mag)', val=1, min=0, max=100)
        self.addWidget('PushButton', 'Dynamic data - average all dynamics for csm', toggle=True, button_title='ON', val=1)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128])
        self.addInPort('coords', 'NPYarray', dtype=[np.float32, np.float64])
        self.addInPort('weights', 'NPYarray', dtype=[np.float32, np.float64])
        self.addInPort('coil sensitivity', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.OPTIONAL)
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)
        self.addOutPort('x', 'NPYarray', dtype=np.complex64)
        self.addOutPort('r', 'NPYarray', dtype=np.complex64)
        self.addOutPort('d', 'NPYarray', dtype=np.complex64)
        self.addOutPort('Applied CSM', 'NPYarray', dtype=np.complex64)
        self.addOutPort('x iterations', 'NPYarray', dtype=np.complex64)

    def validate(self):
        self.log.debug("validate SENSE2")

        iterations = self.getVal('iterations')
        step = self.getVal('step')

        if step and (self.getData('d') is not None):
            # update the UI with the number of iterations
            self.setAttr('iterations', quietval=iterations+1)
        
        # check size of data vs. coords
        self.log.debug("validate SENSE2 - check size of data vs. coords")
        data = self.getData('data')
        coords = self.getData('coords')
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

        self.log.debug("validate SENSE2 - check csm")
        csm = self.getData('coil sensitivity')
        if csm is None:
            self.setAttr('Autocalibration Width (%)', visible=True)
            self.setAttr('Autocalibration Taper (%)', visible=True)
            self.setAttr('Mask Floor (% of max mag)', visible=True)
            if self.getData('data').ndim > 3:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=True)
            else:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
        else:
            self.setAttr('Autocalibration Width (%)', visible=False)
            self.setAttr('Autocalibration Taper (%)', visible=False)
            self.setAttr('Mask Floor (% of max mag)', visible=False)
            self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
            
            # check size of data vs. csm
            if data.ndim != csm.ndim:
                self.log.warn("data and csm do not agree in the number of dimensions")
                return 1
            elif data.shape[:-2] != csm.shape[:-2]:
                self.log.warn("data and csm do not agree in shape (last 2 dimensions don't matter).")
                return 1
                    
        return 0
    

    def compute(self):
        
        self.log.debug("Start CG SENSE 2D")
        # get port and widget inputs
        data = self.getData('data').astype(np.complex64, copy=False)
        coords = self.getData('coords').astype(np.float32, copy=False)
        weights = self.getData('weights').astype(np.float32, copy=False)
        
        mtx_original = self.getVal('mtx')
        iterations = self.getVal('iterations')
        step = self.getVal('step')
        oversampling_ratio = self.getVal('oversampling ratio')
        
        # for a single iteration step use the csm stored in the out port
        if step and (self.getData('Applied CSM') is not None):
            csm = self.getData('Applied CSM')
        else:
            csm = self.getData('coil sensitivity')
        if csm is not None:
            csm = csm.astype(np.complex64, copy=False)
        
        # oversampling: Oversample at the beginning and crop at the end
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
        out_dims_grid = [nr_coils, extra_dim2, extra_dim1, mtx, nr_arms, nr_points]
        out_dims_degrid = [nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points]
        out_dims_fft = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]
        iterations_shape = [extra_dim2, extra_dim1, mtx, mtx]

        # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
        if coords.ndim == 3:
            coords.shape = [1,nr_arms,nr_points,2]

        # output including all iterations
        x_iterations = np.zeros([iterations,extra_dim2,extra_dim1,mtx_original,mtx_original],dtype=np.complex64)
        if step and (iterations > 1):
            previous_iterations = self.getData('x iterations')
            previous_iterations.shape = [iterations-1,extra_dim2, extra_dim1, mtx_original, mtx_original]
            x_iterations[:-1,:,:,:,:] = previous_iterations

        # pre-calculate Kaiser-Bessel kernel
        self.log.debug("Calculate kernel")
        kernel_table_size = 800
        kernel = self.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)
        
        # pre-calculate the rolloff for the spatial domain
        roll = self.rolloff2(mtx, kernel)

        # for a single iteration step use the applied csm and intermediate results stored in outports
        if step and (self.getData('d') is not None):
            # zero-pad csm to oversampled matrix size
            if oversampling_ratio > 1:
                csm = np.pad(csm,[(0,0),(0,0),(0,0),(mtx_min,mtx-mtx_max),(mtx_min,mtx-mtx_max)], 'constant', constant_values=(0,0))
        else: # this is the normal path (not single iteration step)
            # grid to create images that are corrupted by
            # aliasing due to undersampling.  If the k-space data have an
            # auto-calibration region, then this can be used to generate B1 maps.
            self.log.debug("Grid undersampled data")
            gridded_kspace = self.grid2D(data, coords, weights, kernel, out_dims_grid)
            # FFT
            image_domain = self.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
            # rolloff
            image_domain *= roll

            # calculate auto-calibration B1 maps
            if csm is None:
                self.log.debug("Generating autocalibrated B1 maps...")
                csm = self.autocalibrationB1Maps2D(image_domain)
            else:
                # make sure input csm and data are the same mtx size.
                # Assuming the FOV was the same: zero-fill in k-space
                if csm.ndim != 5:
                    self.log.debug("Reshape imported csm")
                    csm.shape = [nr_coils,extra_dim2,extra_dim1,csm.shape[-2],csm.shape[-1]]
                if csm.shape[-1] != mtx:
                    self.log.debug("Interpolate csm to oversampled matrix size")
                    csm_oversampled_mtx = np.int(csm.shape[-1] * oversampling_ratio)
                    if csm_oversampled_mtx%2:
                        csm_oversampled_mtx+=1
                    out_dims_oversampled_image_domain = [nr_coils, extra_dim2, extra_dim1, csm_oversampled_mtx, csm_oversampled_mtx]
                    csm = self.fft2D(csm, dir=1, out_dims_fft=out_dims_oversampled_image_domain)
                    csm = self.fft2D(csm, dir=0, out_dims_fft=out_dims_fft)
            self.setData('Applied CSM', csm[...,mtx_min:mtx_max,mtx_min:mtx_max])

        # keep a conjugate csm set on hand
        csm_conj = np.conj(csm)

        ## Iteration 1:
        if step and (self.getData('d') is not None):
            self.log.debug("\tSENSE Iteration: " + str(iterations))
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
            Ad = self.fft2D(Ad, dir=1)
            Ad = self.degrid2D(Ad, coords, kernel, out_dims_degrid)
            Ad = self.grid2D(Ad, coords, weights, kernel, out_dims_grid)
            Ad = self.fft2D(Ad, dir=0)
            Ad *= roll
            Ad = csm_conj * Ad # broadcast multiply to remove coil phase
            Ad = Ad.sum(axis=0) # assume the coil dim is the first
        else:
            self.log.debug("\tSENSE Iteration: 1")
            # calculate initial conditions
            # d_0
            d_0 = csm_conj * image_domain # broadcast multiply to remove coil phase
            d_0 = d_0.sum(axis=0) # assume the coil dim is the first

            # Ad_0:
            #   degrid -> grid (loop over coils)
            Ad_0 = csm * d_0 # add coil phase
            Ad_0 *= roll # pre-rolloff for degrid convolution
            Ad_0 = self.fft2D(Ad_0, dir=1)
            Ad_0 = self.degrid2D(Ad_0, coords, kernel, out_dims_degrid)
            Ad_0 = self.grid2D(Ad_0, coords, weights, kernel, out_dims_grid)
            Ad_0 = self.fft2D(Ad_0, dir=0)
            Ad_0 *= roll
            Ad_0 = csm_conj * Ad_0 # broadcast multiply to remove coil phase
            Ad_0 = Ad_0.sum(axis=0) # assume the coil dim is the first
            
            # use the initial conditions for the first iter
            r = d = d_0
            x = np.zeros_like(d)
            Ad = Ad_0

        # CG - iter 1
        d_last, r_last, x_last = self.do_cg(d, r, x, Ad)
        
        current_iteration = x_last
        current_iteration.shape = iterations_shape
        if step:
            x_iterations[-1,:,:,:,:] = current_iteration[...,mtx_min:mtx_max,mtx_min:mtx_max]
        else:
            x_iterations[0,:,:,:,:] = current_iteration[...,mtx_min:mtx_max,mtx_min:mtx_max]

        ## Iterations >1:
        for i in range(iterations-1):
            self.log.debug("\tSENSE Iteration: " + str(i+2))

            # input the result of the last iter
            d = d_last
            r = r_last
            x = x_last

            # A
            Ad = csm * d # add coil phase
            Ad *= roll # pre-rolloff for degrid convolution
            Ad = self.fft2D(Ad, dir=1)
            Ad = self.degrid2D(Ad, coords, kernel, out_dims_degrid)
            Ad = self.grid2D(Ad, coords, weights, kernel, out_dims_grid)
            Ad = self.fft2D(Ad, dir=0)
            Ad *= roll
            Ad = csm_conj * Ad # broadcast multiply to remove coil phase
            Ad = Ad.sum(axis=0) # assume the coil dim is the first
            # CG
            d_last, r_last, x_last = self.do_cg(d, r, x, Ad)

            current_iteration = x_last
            current_iteration.shape = iterations_shape
            x_iterations[i+1,:,:,:,:] = current_iteration[..., mtx_min:mtx_max, mtx_min:mtx_max]

        # return the final image     
        self.setData('d', d_last)
        self.setData('r', r_last)
        self.setData('x', x_last)
        self.setData('out', np.squeeze(current_iteration[..., mtx_min:mtx_max, mtx_min:mtx_max]))
        self.setData('x iterations', np.squeeze(x_iterations))

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
        for i in range(int(loops)):
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

    def rolloff2(self, mtx_xy, kernel, clamp_min_percent=10):
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

    def grid2(self, data, mtx_xy, coords, weights, kernel):
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
        return gd.grid(coords, data, weights, kernel, outdim, dx, dy)

    def degrid2(self, gridded_data, coords, kernel):
        import bni.gridding.grid_kaiser as dg
        return dg.degrid(coords, gridded_data, kernel)

    def execType(self):
        # numpy linalg fails if this isn't a thread :(
        #return gpi.GPI_THREAD
        return gpi.GPI_PROCESS

    def autocalibrationB1Maps(self, images):
        # get UI params
        taper = self.getVal('Autocalibration Taper (%)')
        width = self.getVal('Autocalibration Width (%)')
        mask_floor = self.getVal('Mask Floor (% of max mag)')
        average_csm = self.getVal('Dynamic data - average all dynamics for csm')
        
        # Dynamic data - average all dynamics for csm
        if ( (images.ndim > 3) and (average_csm) ):
            nr_dynamics = images.shape[-3]
            images_for_csm = images.sum(axis=-3)
        else:
            images_for_csm = images

        # generate window function for blurring image data
        win = self.window2(images_for_csm.shape[-2:], windowpct=taper, widthpct=width)

        # apply kspace filter
        kspace = self.loop2(self.fft2, images_for_csm, dir=1)
        kspace *= win

        # transform back into image space and normalize
        csm = self.loop2(self.fft2, kspace, dir=0)
        rms = np.sqrt(np.sum(np.abs(csm)**2, axis=0))
        csm = csm / rms

        # zero out points that are below the mask threshold
        thresh = mask_floor/100.0 * rms.max()
        csm *= rms > thresh
        
        # Dynamic data - average all dynamics for csm - asign the average to all dynamics
        if ( (images.ndim > 3) and (average_csm) ):
            out = np.zeros(images.shape, np.complex64)
            for coil in range(images.shape[-4]):
                for dyn in range(nr_dynamics):
                    out[coil,dyn,:,:] = csm[coil,:,:]
        else:
            out=csm

        return out

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
        
        [nr_coils, extra_dim2, extra_dim1, mtx_xy, nr_arms, nr_points] = out_dims
        
        # off-center in pixels.
        dx = dy = 0.

        # gridded kspace
        gridded_kspace = np.zeros([nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy], dtype=data.dtype)
        
        # tell the grid routine what shape to produce
        outdim = np.array([mtx_xy,mtx_xy], dtype=np.int64)

        # coordinate dimensions
        if coords.shape[0] == 1:
            same_coords_for_all_slices_and_dynamics = True
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

    def fft2D(self, data, dir=0, out_dims_fft=[], fft_or_zeropad=True):
        # data: np.complex64
        # dir: int (0 or 1)
        # outdims = [nr_coils, extra_dim2, extra_dim1, mtx, mtx]

        import core.math.fft as corefft

        # generate output dim size array
        # fortran dimension ordering
        if len(out_dims_fft):
            outdims = out_dims_fft.copy()
        else:
            outdims = list(data.shape)
        
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

    def autocalibrationB1Maps2D(self, images):
        # get UI params
        taper = self.getVal('Autocalibration Taper (%)')
        width = self.getVal('Autocalibration Width (%)')
        mask_floor = self.getVal('Mask Floor (% of max mag)')
        average_csm = self.getVal('Dynamic data - average all dynamics for csm')
 
        mtx        = images.shape[-1]
        extra_dim1 = images.shape[-3]
        extra_dim2 = images.shape[-4]
        nr_coils   = images.shape[-5]

        # Dynamic data - average all dynamics for csm
        if ( (extra_dim1 > 1) and (average_csm) ):
            images_for_csm = images.sum(axis=-3)
            images_for_csm.shape = [nr_coils,extra_dim2,1,mtx,mtx]
        else:
            images_for_csm = images

        # generate window function for blurring image data
        win = self.window2(images_for_csm.shape[-2:], windowpct=taper, widthpct=width)

        # apply kspace filter
        kspace = self.fft2D(images_for_csm, dir=1)
        kspace *= win

        # transform back into image space and normalize
        csm = self.fft2D(kspace, dir=0)
        rms = np.sqrt(np.sum(np.abs(csm)**2, axis=0))
        csm = csm / rms

        # zero out points that are below the mask threshold
        thresh = mask_floor/100.0 * rms.max()
        csm *= rms > thresh
        
        # Dynamic data - average all dynamics for csm - asign the average to all dynamics
        if ( (extra_dim1 > 1) and (average_csm) ):
            out = np.zeros(images.shape, np.complex64)
            for coil in range(nr_coils):
                for extra2 in range(extra_dim2):
                    for extra1 in range(extra_dim1):
                        out[coil,extra2,extra1,:,:] = csm[coil,extra2,0,:,:]
        else:
            out=csm

        return out

    def degrid2D(self, data, coords, kernel, outdims):
        # data: np.float32
        # coords: np.complex64
        # weights: np.float32
        # kernel: np.float64
        # outdims = [nr_coils, extra_dim2, extra_dim1, mtx_xy, mtx_xy]: int
        import bni.gridding.grid_kaiser as bni_grid
        
        [nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points] = outdims
        
        # coordinate dimensions
        if coords.shape[0] == 1:
            same_coords_for_all_slices_and_dynamics = True
        else:
            same_coords_for_all_slices_and_dynamics = False

        # gridded kspace
        degridded_kspace = np.zeros([nr_coils, extra_dim2, extra_dim1, nr_arms, nr_points], dtype=data.dtype)
 
        # degrid all slices
        for extra1 in range(extra_dim1):
            if same_coords_for_all_slices_and_dynamics:
                extra1_coords = 0
            else:
                extra1_coords = extra1
            for extra2 in range(extra_dim2):
                for coil in range(nr_coils):
                    degridded_kspace[coil,extra2,extra1,:,:] = bni_grid.degrid(coords[extra1_coords,:,:,:], data[coil,extra2,extra1,:,:], kernel)

        return degridded_kspace

