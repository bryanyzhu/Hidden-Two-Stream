import os, sys
import numpy as np
import cv2

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            #print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (h, w, 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    TAG_CHAR = 'PIEH'    # use this when WRITING the file
    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def norm(arr, axis=-1):
    return np.sqrt(np.sum(arr**2, axis=axis))

def div_nonz(a,b):
    anz = a[b != 0]
    bnz = b[b != 0]
    result = np.zeros_like(a)
    result[b != 0] = anz / bnz
    return result

def flow_ee(f1, f2, mask=None):
	ee_tot = np.sqrt((f1[:,:,:,0] - f2[:,:,:,0])**2 + (f1[:,:,:,1] - f2[:,:,:,1])**2) 
	aee = np.mean(ee_tot, axis=None) 
	# return ee_tot, aee
	return aee

def flow_ae(f1, f2, mask=None):
	u = f1[:,:,:,0]
	u_GT = f2[:,:,:,0]
	v = f1[:,:,:,1]
	v_GT = f2[:,:,:,1]
	numerator = 1 + u * u_GT + v * v_GT
	denominator = np.sqrt(1 + u**2 + v**2) * np.sqrt(1 + u_GT**2 + v_GT**2)
	ae_tot = np.arccos(np.clip(numerator / denominator, -1, 1))
	aae = np.mean(ae_tot, axis=None) 
	# return ae_tot, aae
	return aae

def flowToColor(flow):
	UNKNOWN_FLOW_THRESH = 1e9;
	UNKNOWN_FLOW = 1e10;            

	height, width, nBands = flow.shape

	if nBands != 2:
	    print('flowToColor: flow image must have two bands')    

	u = flow[:,:,0]
	v = flow[:,:,1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999
	maxrad = -1

	# fix unknown flow
	idxUnknown = (abs(u)> UNKNOWN_FLOW_THRESH) | (abs(v)> UNKNOWN_FLOW_THRESH)
	u[idxUnknown] = 0
	v[idxUnknown] = 0

	maxu = np.maximum(maxu, np.amax(u, axis=None))
	minu = np.minimum(minu, np.amin(u, axis=None))

	maxv = np.maximum(maxv, np.amax(v, axis=None))
	minv = np.minimum(minv, np.amin(v, axis=None))

	rad = np.sqrt(u**2 + v**2)
	maxrad = np.maximum(maxrad, np.amax(rad, axis=None))

	# fprintf('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n', maxrad, minu, maxu, minv, maxv);

	# if isempty(varargin) == 0:
	#     maxFlow = varargin{1};
	#     if maxFlow > 0
	#         maxrad = maxFlow;
	#     end;       
	# end;
	eps = 2.22e-16
	u = u / (maxrad + eps)
	v = v / (maxrad + eps)

	# compute color
	img = computeColor(u, v)
	return img 
	    
	# % unknown flow
	# IDX = repmat(idxUnknown, [1 1 3]);
	# img(IDX) = 0;

def computeColor(u,v,logscale=False,scaledown=1,output=False):
    """
    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u**2 + v**2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown    
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot+1)/2 * (ncols-1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)       # 0, 1, 2, ..., ncols

    k1 = k0+1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape+(ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1-f)*col0 + f*col1
       
        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx]*(1-col[idx])
        # out of range    
        col[~idx] *= 0.75
        img[:,:,i] = np.floor(255*col).astype(np.uint8)
    
    return img.astype(np.uint8)
    
def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros((ncols,3))
    
    col = 0
    # RY
    colorwheel[0:RY,0] = 1
    colorwheel[0:RY,1] = np.arange(0,1,1./RY)
    col += RY
    
    # YG
    colorwheel[col:col+YG,0] = np.arange(1,0,-1./YG)
    colorwheel[col:col+YG,1] = 1
    col += YG
    
    # GC
    colorwheel[col:col+GC,1] = 1
    colorwheel[col:col+GC,2] = np.arange(0,1,1./GC)
    col += GC
    
    # CB
    colorwheel[col:col+CB,1] = np.arange(1,0,-1./CB)
    colorwheel[col:col+CB,2] = 1
    col += CB
    
    # BM
    colorwheel[col:col+BM,2] = 1
    colorwheel[col:col+BM,0] = np.arange(0,1,1./BM)
    col += BM
    
    # MR
    colorwheel[col:col+MR,2] = np.arange(1,0,-1./MR)
    colorwheel[col:col+MR,0] = 1

    return colorwheel    
  