#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya", " Sebastian Salazar-Colores", "Abraham Sanchez", "Sebastian Xambò", "Ulises Cortes" ]
__copyright__ = "Copyright 2019, Gobierno de Jalisco"
__credits__ = ["E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["E. Ulises Moya", "Abraham Sanchez"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"


import numpy as np
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
import cv2



def filtergrid(rows, cols):

    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)

    return radius, u1, u2



"""
Function to create to create Riesz kernels
Please see A New Extension of Linear Signal Processing for
Estimating Local Properties and Detecting Features

Get rid of the 0 radius value at the 0 frequency point (at top-left
corner after fftshift) so that taking the log of the radius will not cause trouble.
"""

def riesz_trans(cols, rows):
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)

    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)

    # Compute frequency values as a radius from centre (but quadrant shifted)
    q = np.sqrt(u1 * u1 + u2 * u2)
    q[0, 0] = 1.
    #Construct the monogenic filters in the frequency domain. The two filters
    # would normally be constructed as follows:
    H1= (1j*u1)/q
    H2= (1j*u2)/q
    return H1, H2 



def lowpassfilter(size, cutoff, n):
    """
#Low passfilter in frequency domain
#is useful   
     Constructs a low-pass Butterworth filter:

        f = 1 / (1 + (w/cutoff)^2n)

    usage:  f = lowpassfilter(sze, cutoff, n)

    where:  size    is a tuple specifying the size of filter to construct
            [rows cols].
        cutoff  is the cutoff frequency of the filter 0 - 0.5
        n   is the order of the filter, the higher n is the sharper
            the transition is. (n must be an integer >= 1). Note
            that n is doubled so that it is always an even integer.

    The frequency origin of the returned filter is at the corners.
    """

    if cutoff < 0. or cutoff > 0.5:
        raise Exception('cutoff must be between 0 and 0.5')
    elif n % 1:
        raise Exception('n must be an integer >= 1')
    if len(size) == 1:
        rows = cols = size
    else:
        rows, cols = size

    if (cols % 2):
        xvals = np.arange(-(cols - 1) / 2.,
                          ((cols - 1) / 2.) + 1) / float(cols - 1)
    else:
        xvals = np.arange(-cols / 2., cols / 2.) / float(cols)

    if (rows % 2):
        yvals = np.arange(-(rows - 1) / 2.,
                          ((rows - 1) / 2.) + 1) / float(rows - 1)
    else:
        yvals = np.arange(-rows / 2., rows / 2.) / float(rows)

    x, y = np.meshgrid(xvals, yvals, sparse=True)
    radius = np.sqrt(x * x + y * y)

    return ifftshift(1. / (1. + (radius / cutoff) ** (2. * n)))



'''
nscale          5       Number of wavelet scales, try values 3-6
minWaveLength   3       Wavelength of smallest scale filter.
mult            2.1     Scaling factor between successive filters.
sigmaOnf        0.55    Ratio of the standard deviation of the Gaussian
                        describing the log Gabor filter's transfer function
                        in the frequency domain to the filter center
                        frequency
 Notes on filter settings to obtain even coverage of the spectrum:
    sigmaOnf    .85   mult 1.3
    sigmaOnf    .75   mult 1.6  (filter bandwidth ~1 octave)
    sigmaOnf    .65   mult 2.1
    sigmaOnf    .55   mult 3    (filter bandwidth ~2 octaves)
'''
def logGabor_scale(cols=32,rows=32, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    u1, u2 = np.meshgrid(np.linspace(-0.5, 0.5, cols, endpoint=(cols % 2)),
                         np.linspace(-0.5, 0.5, rows, endpoint=(rows % 2)),
                         sparse=True)
    # Quadrant shift to put 0 frequency at the top left corner
    u1 = ifftshift(u1)
    u2 = ifftshift(u2)
    # Compute frequency values as a radius from centre (but quadrant shifted)
    radius = np.sqrt(u1 * u1 + u2 * u2)
    radius[0, 0] = 1.    
    lp = lowpassfilter((rows, cols), .45, 15) # low pass to cut the maximum size of the log Gabor
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.
    wavelength = minWaveLength * mult**ss
    fo = 1. / wavelength  # Centre frequency of filter
    logRadOverFo = (np.log(radius / fo))
    logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
    logGabor = (lp*logGabor)      # Apply the low-pass filter
    #logGabor[0, 0] = 0. # Undo the radius fudge
    return logGabor



def monogenic_scale(cols=32,rows=32, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    '''
    nscale          5       Number of wavelet scales, try values 3-6
    minWaveLength   3       Wavelength of smallest scale filter.
    mult            2.1     Scaling factor between successive filters.
    sigmaOnf        0.55    Ratio of the standard deviation of the Gaussian
                            describing the log Gabor filter's transfer function
                            in the frequency domain to the filter center
                            frequency
 Notes on filter settings to obtain even coverage of the spectrum:
    sigmaOnf    .85   mult 1.3
    sigmaOnf    .75   mult 1.6  (filter bandwidth ~1 octave)
    sigmaOnf    .65   mult 2.1
    sigmaOnf    .55   mult 3    (filter bandwidth ~2 octaves)
    '''
    
    H1,H2 = riesz_trans(cols, rows)
    logGabor = logGabor_scale(cols,rows, ss,  minWaveLength, mult, sigmaOnf)
    logGabor_H1= logGabor*H1
    logGabor_H2= logGabor*H2
    return logGabor, logGabor_H1, logGabor_H2



def monogenic_filter_one_scale_gray(img, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    # float point is very important in order to compute the fft2
    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype
    # for 3 channel we  make one channel computing the mean value
    if img.ndim == 3:  # hay que cambiar esto
        img = img.mean(2)
    rows, cols = img.shape
    #compute the monnogenic scale in frequency domain
    logGabor, logGabor_H1, logGabor_H2 = monogenic_scale(cols,rows, ss,  minWaveLength, mult, sigmaOnf)
    # FFT2 in the corner
    IM = fft2(img)     # Discrete Fourier Transform of image
    IMF = IM * logGabor   # Frequency bandpassed image
    f = np.real(ifft2(IMF))  # Spatially bandpassed image
    # Bandpassed monogenic filtering, real part of h contains
    IMH1=IM*logGabor_H1
    IMH2=IM*logGabor_H2
    h1= np.real(ifft2(IMH1))
    h2= np.real(ifft2(IMH2))
    # Amplitude of this scale component
    An = np.sqrt(f * f + h1 * h1 + h2 * h2)
    #Orientation computation
    ori = np.arctan(-h2 / h1)
    # Wrap angles between -pi and pi and convert radians to degrees
    ori_d = np.fix((ori % np.pi) / np.pi * 180.)
    # Feature type (a phase angle between -pi/2 and pi/2)
    ft = np.arctan2(f, np.sqrt(h1 * h1 + h2 * h2))
    #proyectionin ij plane
    fr= np.sqrt(h1 * h1 + h2 * h2)
    return An,ori_d,ori,ft,fr,f



def comp_ph_ori_fr_ones_hsv(img, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):

    An,ori_d,ori,ft,fr,f = monogenic_filter_one_scale_gray(img, ss,  minWaveLength, mult, sigmaOnf)
    A_n=cv2.normalize(An, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    ft_n=cv2.normalize(ft, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    ori_n=cv2.normalize(ori, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)   
    fr_n=cv2.normalize(fr, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) 
    f_n=cv2.normalize(f, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) 

    ori_n*=179
    ori_n.astype('uint8')
    ori_n=cv2.convertScaleAbs(ori_n)

    ft_n*=179
    ft_n.astype('uint8')
    ft_n=cv2.convertScaleAbs(ft_n)

    f_n*=255
    f_n.astype('uint8')
    f_n=cv2.convertScaleAbs(f_n)

    fr_n*=255
    fr_n.astype('uint8')
    fr_n=cv2.convertScaleAbs(fr_n)

    ones = np.zeros(A_n.shape)
    ones[:,:]=255
    ones.astype('uint8')
    ones =cv2.convertScaleAbs(ones)

    n_img1 =cv2.merge((ft_n, fr_n, ones))

    n_img2 =cv2.merge((ori_n,fr_n, ones))

    return n_img1,n_img2



def comp_ph_ori_ones_ones_hsv(img, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):

    An,ori_d,ori,ft,fr,f = monogenic_filter_one_scale_gray(img, ss,  minWaveLength, mult, sigmaOnf)
    A_n=cv2.normalize(An, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    ft_n=cv2.normalize(ft, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    ori_n=cv2.normalize(ori, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)   
    fr_n=cv2.normalize(fr, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) 
    f_n=cv2.normalize(f, None, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F) 

    ori_n*=179
    ori_n.astype('uint8')
    ori_n=cv2.convertScaleAbs(ori_n)

    ft_n*=179
    ft_n.astype('uint8')
    ft_n=cv2.convertScaleAbs(ft_n)

    f_n*=255
    f_n.astype('uint8')
    f_n=cv2.convertScaleAbs(f_n)

    fr_n*=255
    fr_n.astype('uint8')
    fr_n=cv2.convertScaleAbs(fr_n)

    ones = np.zeros(A_n.shape)
    ones[:,:]=255
    ones.astype('uint8')
    ones =cv2.convertScaleAbs(ones)

    n_img1 =cv2.merge((ft_n, ones, ones))

    n_img2 =cv2.merge((ori_n,ones, ones))

    return n_img1,n_img2



def comp_ph_ori_fr_ones_rgb(img, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    phase,ori=comp_ph_ori_fr_ones_hsv(img, ss,  minWaveLength, mult, sigmaOnf)
    if img.ndim == 3:  # it possible to improve this part by computing as input each channel
        img = img.mean(2)
    rows, cols = img.shape
    a=rows
    b=cols
    
    bgr1 = cv2.cvtColor(phase, cv2.COLOR_HSV2BGR)
    phase_rgb =cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    
    bgr2 = cv2.cvtColor(ori, cv2.COLOR_HSV2BGR)
    ori_rgb =cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    ch_num = 6
    ph_ori_ones6 = np.zeros((a,b,ch_num),dtype='uint8')
    ph_ori_ones6[:,:,0]= phase_rgb[:,:,0]
    ph_ori_ones6[:,:,1]= phase_rgb[:,:,1]
    ph_ori_ones6[:,:,2]= phase_rgb[:,:,2]
    ph_ori_ones6[:,:,3]= ori_rgb[:,:,0]
    ph_ori_ones6[:,:,4]= ori_rgb[:,:,1]
    ph_ori_ones6[:,:,5]= ori_rgb[:,:,2]

    return ph_ori_ones6



def comp_ph_ori_fr_ones_rgb_list(input_list, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    x3= []
    for i in range(0,len(input_list)):
        x2=comp_ph_ori_fr_ones_rgb(input_list[i], ss,  minWaveLength, mult, sigmaOnf)
        x3.append(x2)
    return x3



def comp_ph_ori_ones_ones_rgb(img, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    phase,ori=comp_ph_ori_ones_ones_hsv(img, ss,  minWaveLength, mult, sigmaOnf)
    
    if img.ndim == 3:  # it possible to improve this part by computing as input each channel
        img = img.mean(2)
    rows, cols = img.shape
    a=rows
    b=cols
    bgr1 = cv2.cvtColor(phase, cv2.COLOR_HSV2BGR)
    phase_rgb =cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    
    bgr2 = cv2.cvtColor(ori, cv2.COLOR_HSV2BGR)
    ori_rgb =cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

    ch_num = 6
    ph_ori_ones6 = np.zeros((a,b,ch_num),dtype='uint8')
    ph_ori_ones6[:,:,0]= phase_rgb[:,:,0]
    ph_ori_ones6[:,:,1]= phase_rgb[:,:,1]
    ph_ori_ones6[:,:,2]= phase_rgb[:,:,2]
    ph_ori_ones6[:,:,3]= ori_rgb[:,:,0]
    ph_ori_ones6[:,:,4]= ori_rgb[:,:,1]
    ph_ori_ones6[:,:,5]= ori_rgb[:,:,2]
    return ph_ori_ones6



def comp_ph_ori_ones_ones_rgb_list(input_list, ss=1,  minWaveLength=3, mult=2.1, sigmaOnf=0.55):
    x3= []
    for i in range(0,len(input_list)):
        x2=comp_ph_ori_ones_ones_rgb(input_list[i], ss,  minWaveLength, mult, sigmaOnf)
        x3.append(x2)
    return x3



def circle(size, cutoff, n):
    '''
    binary circle image
    '''
    lp = lowpassfilter(size, cutoff, n)
    circle = np.fft.fftshift(lp)

    return circle    



def line(size, lenth, theta):
    '''
    binary rotated line or bar image
    '''
    rows, cols = size #have to be even and equal
    #lenth #have to be odd
    img = np.zeros([rows,cols], dtype=np.uint8)
    r4= rows/4
    ry0 = r4
    ry1 = 3*r4
    rx0 = 2*r4 - lenth/2
    rx1 = 2*r4 + lenth/2
    img[int(ry0):int(ry1),int(rx0):int(rx1)]= 255
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    im = cv2.warpAffine(img,M,(cols,rows))    
    return im


