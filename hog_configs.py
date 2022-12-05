### HOG Feature Descriptor Configurations
# check out: https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html#a847f3d42f1cf72897d5263fe3217a36d
# for a description on parameters

winSize = (64,64)
blockSize = (16,16)
blockStride= (8,8)
cellSize = (8,8)

nbins = 9
derivAperture = 1
winSigma = 4.0
histogramNormType = 0

L2HysThreshold = 2e-1
gammaCorrection = 0
nlevels = 64

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

# I want to try using configargparse next time
