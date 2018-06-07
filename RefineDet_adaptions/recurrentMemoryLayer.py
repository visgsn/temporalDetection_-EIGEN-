'''
    This layer implements a recurrent shift register which is able to pass information from one
    net iteration (time step) to another

    Usage:
'''

import caffe



##### CONFIGURATION ####################################################################################################
TRAIN   = 0
TEST    = 1
########################################################################################################################



class recurrent_memory_layer(caffe.Layer):
    def setup(self, bottom, top):
        # Check top shape
        if len(top) != 1:
            raise Exception("Need to define top blob (data).")

        # Check bottom shape
        if len(bottom) != 1:
            raise Exception("Need to define bottom blob (data).")

        # Read parameters
        params = eval(self.param_str)
        src_file = params["src_file"]
        self.batch_size = params["batch_size"]
        self.im_shape = params["im_shape"]
        self.crop_size = params.get("crop_size", False)

        ###### Reshape top ######
        # This could also be done in Reshape method, but since it is a one-time-only
        # adjustment, we decided to do it on Setup
        if self.crop_size:
            top[0].reshape(self.batch_size, 3, self.crop_size, self.crop_size)
        else:
            top[0].reshape(self.batch_size, 3, self.im_shape, self.im_shape)

        top[1].reshape(self.batch_size)

        # Read source file
        # I'm just assuming we have this method that reads the source file
        # and returns a list of tuples in the form of (img, label)
        self.imgTuples = readSrcFile(src_file)

        self._cur = 0  # use this to check if we need to restart the list of imgs


    def forward(self, bottom, top):
        pass


    def reshape(self, bottom, top):
        pass


    def backward(self, bottom, top):
        pass

