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
        self.addWidget('SpinBox', 'number of threads', val=4, min=1, max=64, collapsed=True)
        self.addWidget('Slider', 'Autocalibration Width (%)', val=15, min=0, max=100)
        self.addWidget('Slider', 'Autocalibration Taper (%)', val=50, min=0, max=100)
        self.addWidget('Slider', 'Mask Floor (% of max mag)', val=10, min=0, max=100)
        self.addWidget('SpinBox', 'Autocalibration mask dilation [pixels]', val=10, min=0, max=100)
        self.addWidget('SpinBox','Autocalibration SDC Iterations',val=10, min=1)
        self.addWidget('PushButton', 'Dynamic data - average all dynamics for csm', toggle=True, button_title='ON', val=1)
        self.addWidget('PushButton', 'Golden Angle - combine dynamics before gridding', toggle=True, button_title='ON', val=1)
        self.addWidget('Slider', '# golden angle dynamics for csm', val=150, min=0, max=10000)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128])
        self.addInPort('coords', 'NPYarray', dtype=[np.float32, np.float64])
        self.addInPort('weights', 'NPYarray', dtype=[np.float32, np.float64])
        self.addInPort('coil sensitivity', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.OPTIONAL)
        self.addOutPort('out', 'NPYarray', dtype=np.complex64)
        self.addOutPort('x', 'NPYarray', dtype=np.complex64)
        self.addOutPort('r', 'NPYarray', dtype=np.complex64)
        self.addOutPort('d', 'NPYarray', dtype=np.complex64)
        self.addOutPort('oversampled CSM', 'NPYarray', dtype=np.complex64)
        self.addOutPort('cropped CSM', 'NPYarray', dtype=np.complex64)
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
                    self.log.warn("data and coords do not agree in the number of phases / dynamics: coords.shape[-4]="+str(coords.shape[-4])+" != data.shape[-3]="+str(data.shape[-3]))
                    return 1

        self.log.debug("validate SENSE2 - check csm")
        csm = self.getData('coil sensitivity')
        if csm is None:
            self.setAttr('Autocalibration Width (%)', visible=True)
            self.setAttr('Autocalibration Taper (%)', visible=True)
            self.setAttr('Mask Floor (% of max mag)', visible=True)
            self.setAttr('Autocalibration mask dilation [pixels]', visible=True)
            self.setAttr('Autocalibration SDC Iterations', visible=True)
            if self.getData('data').ndim > 3:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=True)
            else:
                self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
        else:
            self.setAttr('Autocalibration Width (%)', visible=False)
            self.setAttr('Autocalibration Taper (%)', visible=False)
            self.setAttr('Mask Floor (% of max mag)', visible=False)
            self.setAttr('Autocalibration mask dilation [pixels]', visible=False)
            self.setAttr('Dynamic data - average all dynamics for csm', visible=False)
            self.setAttr('Autocalibration SDC Iterations', visible=False)
            
            # check size of data vs. csm
            if data.shape[0] != csm.shape[0]:
                self.log.warn("data and csm do not agree in in number of coils.")
                return 1
        GA = self.getVal('Golden Angle - combine dynamics before gridding')
        if GA:
            self.setAttr('# golden angle dynamics for csm', visible=True)
            if coords.ndim == 3:
                self.setAttr('# golden angle dynamics for csm', max=coords.shape[-3])
            elif coords.ndim == 4:
                self.setAttr('# golden angle dynamics for csm', max=coords.shape[-3]*coords.shape[-4])
        else:
            self.setAttr('# golden angle dynamics for csm', visible=False)
        return 0
    

    def compute(self):
        import bni.gridding.Kaiser2D_utils as kaiser2D
        
        self.log.debug("Start CG SENSE 2D")
        # get port and widget inputs
        data = self.getData('data').astype(np.complex64, copy=False)
        coords = self.getData('coords').astype(np.float32, copy=False)
        weights = self.getData('weights').astype(np.float32, copy=False)
        
        mtx_original = self.getVal('mtx')
        iterations = self.getVal('iterations')
        step = self.getVal('step')
        oversampling_ratio = self.getVal('oversampling ratio')
        number_threads = self.getVal('number of threads')
        GA = self.getVal('Golden Angle - combine dynamics before gridding')
        
        # for a single iteration step use the csm stored in the out port
        if step and (self.getData('oversampled CSM') is not None):
            csm = self.getData('oversampled CSM')
        else:
            csm = self.getData('coil sensitivity')
        if csm is not None:
            csm = csm.astype(np.complex64, copy=False)
        
        # oversampling: Oversample at the beginning and crop at the end
        mtx = np.int(np.around(mtx_original * oversampling_ratio))
        if mtx%2:
            mtx+=1
        if oversampling_ratio > 1:
            mtx_min = np.int(np.around((mtx-mtx_original)/2))
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
            weights.shape = [1,nr_arms,nr_points]

        # output including all iterations
        x_iterations = np.zeros([iterations,extra_dim2,extra_dim1,mtx_original,mtx_original],dtype=np.complex64)
        if step and (iterations > 1):
            previous_iterations = self.getData('x iterations')
            previous_iterations.shape = [iterations-1,extra_dim2, extra_dim1, mtx_original, mtx_original]
            x_iterations[:-1,:,:,:,:] = previous_iterations

        # pre-calculate Kaiser-Bessel kernel
        self.log.debug("Calculate kernel")
        kernel_table_size = 800
        kernel = kaiser2D.kaiserbessel_kernel( kernel_table_size, oversampling_ratio)
        
        # pre-calculate the rolloff for the spatial domain
        
        roll = kaiser2D.rolloff2D(mtx, kernel)

        # for a single iteration step use the oversampled csm and intermediate results stored in outports
        if step and (self.getData('d') is not None):
            self.log.debug("Save some time and use the previously determined csm stored in the cropped CSM outport.")
        elif GA: #combine data from GA dynamics before gridding, use code from VirtualChannels_GPI.py
            # grid images for each phase - needs to be done at some point, not really here for csm though.
            self.log.debug("Grid undersampled data")
            gridded_kspace = kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            # FFT
            image_domain = kaiser2D.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
            # rolloff
            image_domain *= roll
            
            twoD_or_threeD = coords.shape[-1]
            # parameters from UI
            UI_width = self.getVal('Autocalibration Width (%)')
            UI_taper = self.getVal('Autocalibration Taper (%)')
            UI_mask_floor = self.getVal('Mask Floor (% of max mag)')
            mask_dilation = self.getVal('Autocalibration mask dilation [pixels]')
            UI_average_csm = self.getVal('Dynamic data - average all dynamics for csm')
            numiter = self.getVal('Autocalibration SDC Iterations')
            original_csm_mtx = np.int(0.01 * UI_width * mtx_original)
            
            is_GoldenAngle_data = True
            nr_arms_csm = self.getVal('# golden angle dynamics for csm')
            csm_data = data.copy()
            nr_all_arms_csm = extra_dim1 * nr_arms
            #extra_dim2_csm = 1
            extra_dim1_csm = 1
            #csm_data.shape = [nr_coils,extra_dim2_csm,extra_dim1_csm,nr_all_arms_csm,nr_points]
            #csm_data = csm_data[:,0:nr_arms_csm,:]
            
            # coords dimensions: (add 1 dimension as they could have another dimension for golden angle dynamics
            if coords.ndim == 3:
                coords.shape = [1,nr_arms,nr_points,twoD_or_threeD]
            
            # create low resolution csm
            # cropping the data will make gridding and FFT much faster
            magnitude_one_interleave = np.zeros(nr_points)
            for x in range(nr_points):
                magnitude_one_interleave[x] = np.sqrt( coords[0,0,x,0]**2 + coords[0,0,x,1]**2)
            within_csm_width_radius = magnitude_one_interleave[:] < (0.01 * UI_width * 0.5) # for BNI spirals should be 0.45 instead of 0.5
            nr_points_csm_width = within_csm_width_radius.sum()
            # in case of radial trajectory, it doesn't start at zero..
            found_start_point = 0
            found_end_point = 0
            for x in range(nr_points):
                if ((not found_start_point) and (within_csm_width_radius[x])):
                    found_start_point = 1
                    start_point = x
                if ((not found_end_point) and (found_start_point) and (not within_csm_width_radius[x])):
                    found_end_point = 1
                    end_point = x
            if not found_end_point:
                end_point = nr_points
            self.log.node("Start and end points in interleave are: "+str(start_point)+" and "+str(end_point)+" leading to "+str(nr_points_csm_width)+" points for csm.")
            
            arm_counter = 0
            extra_dim1_counter = 0
            arm_with_data_counter = 0
            while (arm_with_data_counter < nr_arms_csm and extra_dim1_counter < extra_dim1):
                if (coords[extra_dim1_counter, arm_counter,0,0] != coords[extra_dim1_counter, arm_counter,-1,0]): #only equal when no data in this interleave during resorting
                    arm_with_data_counter += 1
                arm_counter += 1
                if arm_counter == nr_arms:
                    arm_counter = 0
                    extra_dim1_counter += 1
            self.log.node("Found "+str(arm_with_data_counter)+" arms, and was looking for "+str(nr_arms_csm)+" from a total of "+str(nr_all_arms_csm)+" arms.")
            
            csm_data = np.zeros([nr_coils,extra_dim2,extra_dim1_csm,arm_with_data_counter,nr_points_csm_width], dtype=data.dtype)
            csm_coords = np.zeros([1,arm_with_data_counter,nr_points_csm_width,twoD_or_threeD], dtype=coords.dtype)
            
            arm_counter = 0
            extra_dim1_counter = 0
            arm_with_data_counter = 0
            while (arm_with_data_counter < nr_arms_csm and extra_dim1_counter < extra_dim1):
                if (coords[extra_dim1_counter, arm_counter,0,0] != coords[extra_dim1_counter, arm_counter,-1,0]): #only equal when no data in this interleave during resorting
                    csm_data[:,:,0,arm_with_data_counter,:] = data[:,:,extra_dim1_counter,arm_counter,start_point:end_point]
                    csm_coords[0,arm_with_data_counter,:,:] = coords[extra_dim1_counter,arm_counter,start_point:end_point,:]
                    arm_with_data_counter += 1
                arm_counter += 1
                if arm_counter == nr_arms:
                    arm_counter = 0
                    extra_dim1_counter += 1
            self.log.node("Found "+str(arm_with_data_counter)+" arms, and was looking for "+str(nr_arms_csm)+" from a total of "+str(nr_all_arms_csm)+" arms.")
            
            # now set the dimension lists
            out_dims_grid_csm = [nr_coils, extra_dim2, extra_dim1_csm, mtx, arm_with_data_counter, nr_points_csm_width]
            out_dims_fft_csm = [nr_coils, extra_dim2, extra_dim1_csm, mtx, mtx]
            
            # generate SDC based on number of arms and nr of points being used for csm
            import core.gridding.sdc as sd
            #csm_weights = sd.twod_sdcsp(csm_coords.squeeze().astype(np.float64), numiter, 0.01 * UI_taper, mtx)
            cmtxdim = np.array([mtx,mtx],dtype=np.int64)
            wates = np.ones((arm_with_data_counter * nr_points_csm_width), dtype=np.float64)
            coords_for_sdc = csm_coords.astype(np.float64)
            coords_for_sdc.shape = [arm_with_data_counter * nr_points_csm_width, twoD_or_threeD]
            csm_weights = sd.twod_sdc(coords_for_sdc, wates, cmtxdim, numiter, 0.01 * UI_taper )
            csm_weights.shape = [1,arm_with_data_counter,nr_points_csm_width]
            
            # Grid
            gridded_kspace_csm = kaiser2D.grid2D(csm_data, csm_coords, csm_weights.astype(np.float32), kernel, out_dims_grid_csm, number_threads=number_threads)
            image_domain_csm = kaiser2D.fft2D(gridded_kspace_csm, dir=0, out_dims_fft=out_dims_fft_csm)
            # rolloff
            image_domain_csm *= roll
            # # crop to original matrix size
            # csm = image_domain_csm[...,mtx_min:mtx_max,mtx_min:mtx_max]
            # normalize by rms (better would be to use a whole body coil image
            csm_rms = np.sqrt(np.sum(np.abs(image_domain_csm)**2, axis=0))
            image_domain_csm = image_domain_csm / csm_rms
            # zero out points that are below mask threshold
            thresh = 0.01 * UI_mask_floor * csm_rms.max()
            mask = csm_rms > thresh
            # use scipy library to grow mask and fill holes.
            from scipy import ndimage
            mask.shape = [mtx,mtx]
            mask = ndimage.morphology.binary_dilation(mask, iterations=mask_dilation)
            mask = ndimage.binary_fill_holes(mask)
            
            image_domain_csm *= mask
            if extra_dim1 > 1:
                csm = np.zeros([nr_coils, extra_dim2, extra_dim1, mtx, mtx], dtype=image_domain_csm.dtype)
                for extra_dim1_counter in range(extra_dim1):
                    csm[:,:,extra_dim1_counter,:,:]=image_domain_csm[:,:,0,:,:]
            else:
                csm = image_domain_csm
            self.setData('oversampled CSM', csm)
            self.setData('cropped CSM', csm[...,mtx_min:mtx_max,mtx_min:mtx_max])
            
        else: # this is the normal path (not single iteration step)
            # grid to create images that are corrupted by
            # aliasing due to undersampling.  If the k-space data have an
            # auto-calibration region, then this can be used to generate B1 maps.
            self.log.debug("Grid undersampled data")
            gridded_kspace = kaiser2D.grid2D(data, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            # FFT
            image_domain = kaiser2D.fft2D(gridded_kspace, dir=0, out_dims_fft=out_dims_fft)
            # rolloff
            image_domain *= roll

            # calculate auto-calibration B1 maps
            if csm is None:
                self.log.debug("Generating autocalibrated B1 maps...")
                # parameters from UI
                UI_width = self.getVal('Autocalibration Width (%)')
                UI_taper = self.getVal('Autocalibration Taper (%)')
                UI_mask_floor = self.getVal('Mask Floor (% of max mag)')
                UI_average_csm = self.getVal('Dynamic data - average all dynamics for csm')
                csm = kaiser2D.autocalibrationB1Maps2D(image_domain, taper=UI_taper, width=UI_width, mask_floor=UI_mask_floor, average_csm=UI_average_csm)
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
                    csm = kaiser2D.fft2D(csm, dir=1, out_dims_fft=out_dims_oversampled_image_domain)
                    csm = kaiser2D.fft2D(csm, dir=0, out_dims_fft=out_dims_fft)
            self.setData('oversampled CSM', csm)
            self.setData('cropped CSM', csm[...,mtx_min:mtx_max,mtx_min:mtx_max])

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
            Ad = kaiser2D.fft2D(Ad, dir=1)
            Ad = kaiser2D.degrid2D(Ad, coords, kernel, out_dims_degrid, number_threads=number_threads, oversampling_ratio = oversampling_ratio)
            Ad = kaiser2D.grid2D(Ad, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            Ad = kaiser2D.fft2D(Ad, dir=0)
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
            Ad_0 = kaiser2D.fft2D(Ad_0, dir=1)
            Ad_0 = kaiser2D.degrid2D(Ad_0, coords, kernel, out_dims_degrid, number_threads=number_threads, oversampling_ratio = oversampling_ratio)
            Ad_0 = kaiser2D.grid2D(Ad_0, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            Ad_0 = kaiser2D.fft2D(Ad_0, dir=0)
            Ad_0 *= roll
            Ad_0 = csm_conj * Ad_0 # broadcast multiply to remove coil phase
            Ad_0 = Ad_0.sum(axis=0) # assume the coil dim is the first
            
            # use the initial conditions for the first iter
            r = d = d_0
            x = np.zeros_like(d)
            Ad = Ad_0

        # CG - iter 1 or step
        d_last, r_last, x_last = self.do_cg(d, r, x, Ad)
        
        current_iteration = x_last.copy()
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
            Ad = kaiser2D.fft2D(Ad, dir=1)
            Ad = kaiser2D.degrid2D(Ad, coords, kernel, out_dims_degrid, number_threads=number_threads, oversampling_ratio = oversampling_ratio)
            Ad = kaiser2D.grid2D(Ad, coords, weights, kernel, out_dims_grid, number_threads=number_threads)
            Ad = kaiser2D.fft2D(Ad, dir=0)
            Ad *= roll
            Ad = csm_conj * Ad # broadcast multiply to remove coil phase
            Ad = Ad.sum(axis=0) # assume the coil dim is the first
            # CG
            d_last, r_last, x_last = self.do_cg(d, r, x, Ad)

            current_iteration = x_last.copy()
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

    def execType(self):
        # numpy linalg fails if this isn't a thread :(
        #return gpi.GPI_THREAD
        return gpi.GPI_PROCESS
