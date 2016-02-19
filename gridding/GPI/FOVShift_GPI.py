# Author: Nick Zwart
# Date: 2016feb18

import numpy as np
import gpi

# This is a template node, with stubs for initUI() (input/output ports,
# widgets), validate(), and compute().
# Documentation for the node API can be found online:
# http://docs.gpilab.com/NodeAPI

class ExternalNode(gpi.NodeAPI):
    """FOVShift uses the coordinates and input shift arguments to add a linear
    phase to the data.  Currently this only supports 2D data.

    INPUT:
        crds - 2-vec array
        data - raw k-space data corresponding to crds
    OUTPUT:
        adjusted data - original k-space data with linear phase
    WIDGETS:
        dx,dy - FOV shift in pixels
    """

    # initialize the UI - add widgets and input/output ports
    def initUI(self):
        # Widgets
        self.addWidget('DoubleSpinBox', 'dx (pixels)', val=0)
        self.addWidget('DoubleSpinBox', 'dy (pixels)', val=0)

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64,np.complex128])
        self.addInPort('crds', 'NPYarray', dtype=[np.float32,np.float64], vec=2)
        self.addOutPort('adjusted data', 'NPYarray', dtype=[np.complex64,np.complex128])


    # process the input data, send it to the output port
    # return 1 if the computation failed
    # return 0 if the computation was successful 
    def compute(self):
        data = self.getData('data')
        crds = self.getData('crds')
        dx = self.getVal('dx (pixels)')
        dy = self.getVal('dy (pixels)')

        phase = np.exp(-1j * 2.0 * np.pi * (crds[...,0]*dx + crds[...,1]*dy))

        self.setData('adjusted data', data * phase)

        return 0
