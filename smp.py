import numpy as np 
import astropy.table as tb 
import astropy.io.fits as  pf 
import compress_pickle as cp
import scipy.sparse as sp  

import piff 
from pixmappy import DESMaps, Gnomonic 

from scipy.linalg import block_diag, lstsq
from numpy.linalg import LinAlgError

gain_shape = np.zeros((4096, 2048))



def radec2point(RA, DEC, filt):
    #This function takes in RA and DEC and returns the pointing and SCA with
    #center closest to desired RA/DEC
    f = fits.open('/cwork/mat90/RomanDESC_sims_2024/RomanTDS/Roman_TDS_obseq_11_6_23_radec.fits')[1]
    f = f.data
    allRA = f['RA']
    allDEC = f['DEC']
    dist = np.sqrt((allRA[:,:,np.newaxis] - RA)**2 + (allDEC[:,:,np.newaxis] - DEC)**2)
    dist[np.where(f['filter'] != filt)] = np.inf #Ensuring we only get the filter we want

    reshaped_array = dist.reshape(-1, dist.shape[2])
    # Find the indices of the minimum values along the flattened slices
    min_indices = np.argmin(reshaped_array, axis=0)
    # Convert the flat indices back to 2D coordinates
    rows, cols = np.unravel_index(min_indices, dist.shape[:2])
    # Combine rows and cols into a list of tuples for coordinates
    coords = np.stack((rows, cols), axis=1)

    #coords = np.unravel_index(dist.argmin(), dist.shape) #Returns the index of the minimum value in the array
    x_coords, y_coords = coords[:, 0], coords[:, 1]
    if len(x_coords) == 1:
        x_coords = x_coords[0]
        y_coords = y_coords[0]



    return x_coords, y_coords + 1

def downsample(array,factor):
    """
    Downsample an array by a factor of `factor` in each dimension
    """
    xsize = array.shape[0]
    ysize = array.shape[1]
    output = np.zeros((xsize//factor,ysize//factor))
    assert xsize % factor == 0 and ysize % factor == 0, "Array size must be divisible by factor size is " + str(xsize) + " " + str(ysize) + " factor is " + str(factor)
    for i in range(xsize//factor):
        for j in range(ysize//factor):
            output[i,j] = np.mean(array[i*factor:(i+1)*factor + 1, j*factor:(j+1)*factor + 1]) #I am unclear if this +1 is correct XXX TODO

    return output

def gain(expnum, ccdnum, x, y, stampsize, gain_dict): #Unedited PEDRO code
    '''
    Uses the gain table to construct a stamp that is reproduces the gain in the original image, and
    then cuts the image out to reproduce the stamp
    '''
    from astropy.nddata import Cutout2D

    gain_exp = gain_dict[expnum][ccdnum]
    gain_image = gain_shape

    if ccdnum < 32:
        gain_image[:,:1024] = gain_exp['B']
        gain_image[:,1024:] = gain_exp['A']
    else:
        gain_image[:,:1024] = gain_exp['A']
        gain_image[:,1024:] = gain_exp['B']


    gain_cut = Cutout2D(gain_image, (x,y), stampsize,  mode='partial', fill_value=0)

    return gain_cut.data




def construct_psf_background(ra, dec, wcs, psf, x_loc, y_loc, stampsize, flatten = True, color=0.61): #My version
    '''
    Constructs the background model using PIFF's PSFs around a certain image (x,y) location and a given array of RA and DECs.
    The pixel coordinates are found using pixmappy's WCSs 
    stampsize determines how large the image will be (eg stampsize = 30 means a 30x30 image). 
    flatten decides if the image should be flattened (preferred) or not
    '''
    print('Constructing psf background...')
    
  
    osample = 8

    x, y = wcs.world_to_pixel(SkyCoord(ra = np.array(ra)*u.degree, dec = np.array(dec)*u.degree))
    x_center = np.median(x)
    y_center = np.median(y)



    if type(x_loc) == np.ndarray and np.size(x_loc) > 1:
        print(x_loc, 'x_loc')
        print(np.size(x_loc), 'size of x_loc')
        print(y_loc, 'y_loc')
        x_loc = x_loc[0]
        y_loc = y_loc[0]


    config_file = '/hpc/home/cfm37/my_tds.yaml' 

    xdists = 2*np.abs(x_loc - np.rint(x).astype(int))
    ydists = 2*np.abs(y_loc - np.rint(y).astype(int))
    alldists = np.concatenate((xdists, ydists))
    
    bonussize = stampsize * 2
    #print(bonussize, 'bonussize')
    total = stampsize + bonussize
    #print(total, 'total')
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=scanum)
    master = util_ref.getPSF_Image(total, x=x_loc, y=y_loc, oversampling_factor = osample).array 
    center = osample*total//2
    #print(np.shape(master), 'master shape')


    x_over = x * osample
    y_over = y * osample #Oversampling by a factor of 8
    
    x_over = np.rint(x_over).astype(int)
    y_over = np.rint(y_over).astype(int)


    deltax = x_over - x_center *osample
    deltay = y_over - y_center *osample
    psfs = np.zeros((31 * 31,np.size(deltax)))

    k = 0
    for i,j in zip(deltax.flatten(),deltay.flatten()):
        cutout = Cutout2D(master, (center-i + osample//2, center-j+osample//2), stampsize*osample, mode = 'strict') #I am not sure why this +osample//2 is necessary, need to figure it out before using
        
        
        down = downsample(cutout.data,osample)

        psfs[:,k] = down.flatten()
        k += 1
        '''
        if k % 11 == 0:
            plt.subplot(1,2,1)
            plt.imshow(master)
            cutout.plot_on_original(color='red')
            plt.subplot(1,2,2)
            plt.imshow(down, vmin = 0, vmax = 0.0029)
            plt.show()
        '''

    #plt.imshow(np.sum(psfs, axis = 1).reshape(31,31), origin = 'lower')
    #plt.show()

    return psfs


def construct_psf_source(x, y, psf, stampsize=30, x_center = None, y_center = None, color=0.61): #My version
    '''
        Constructs the PIFF PSF around the point source (x,y) location, allowing for some offset from the center
         (if so, specify x_center and y_center)

    '''

    if x_center is None:
        print('Setting x_center to x')
        x_center = x 
        y_center = y
    else:
        print('x_center has a specific value')
        
    if type(x_center) == np.ndarray and np.size(x_center) > 1:
        x_center = x_center[0]
        y_center = y_center[0]
    if type(x) == np.ndarray:
        x = x[0]
        y = y[0]
    #Need to customize band stuff here too XXX TODO 
    config_file = '/hpc/home/cfm37/my_tds.yaml'
    util_ref = roman_utils(config_file=config_file, visit = pointing, sca=SCA)
    print('Placing PSF at x,y:', x, y)
    print('Rounded x,y:', np.rint(x).astype(int), np.rint(y).astype(int))
    shiftx = (x - np.rint(x).astype(int))
    shifty = (y - np.rint(y).astype(int))
    print('shifting by:', shiftx, shifty)


    osample = 12
    master = util_ref.getPSF_Image(stampsize + 2*osample, x=x, y=y, oversampling_factor = osample).array
    center = np.shape(master)[0]//2 #added parentheses
    #print(center, 'center pixel in big image')
    i = shiftx * osample
    j = shifty * osample
    #print('new center:', center - i, center - j)
    cutout = Cutout2D(master, (center - i, center - j), stampsize*osample, mode = 'strict') #Not sure why the osample//2 is necessary, removed it for now?? 
    down = downsample(cutout.data,osample)
      
    return down.flatten()


def local_grid(ra_center, dec_center, step, npoints):
    '''
    Generates a local grid around a RA-Dec center, choosing step size and number of points
    '''

    x = np.linspace(-step*npoints/2, step*npoints/2, npoints)
    y = np.linspace(-step*npoints/2, step*npoints/2, npoints)

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    ra_grid, dec_grid = Gnomonic(ra_center, dec_center).toSky(xx, yy)

    return ra_grid, dec_grid




class Detection:
    '''
    Main class for SMP. Requires RA and Dec for the detection, an exposure and CCD numbers for bookkeeping and
    zero-point retrieval, a band (for finding extra exposures) and an optional color (for astrometry) and name for the 
    detection 
    '''
    def __init__(self, ra, dec, expnum, ccdnum, band, color = 0.61, name = ''):
        '''
        Constructor class
        '''
        self.ra = ra 
        self.dec = dec 
        self.expnum = expnum
        self.ccdnum = ccdnum 
        self.band = band
        self.color = color
        self.name = name

    def write(self, filename):
        '''
        Saves a pickle file (with a given filename) for the detection, can be reloaded back with the read function
        '''
        cp.dump(self, filename)

    @staticmethod
    def read(filename):
        '''	
        Loads a previously saved detection. Usage:
            det = Detection.read(filename)
        '''
        return cp.load(filename)


    def findAllExposures(self, survey, return_list = False, reduce_band = True):
        '''
        Requires a list of DECamExposures or DESExposures from `DESTNOSIM`, 
        returns all exposures that touch the point. 
        If return_list == True, returns this as a list, otherwise saves this 
        inside det.exposures
        If reduce_band == True, drops all exposures from different bands (that is,
        keeps only the same band as the detection)
        '''
        from pixmappy import Gnomonic
        x, y = Gnomonic(self.ra, self.dec).toXY(np.array(survey.ra), np.array(survey.dec))
        #note that, at RA0, DEC0, x,y = 0
        dist = np.sqrt(x**2 + y**2)

        close = np.where(dist < 1.5)
        ra_arr = np.array([self.ra])
        dec_arr = np.array([self.dec])

        ccdlist = tb.Table(names=('EXPNUM', 'CCDNUM', 'BAND'), dtype=('i8', 'i4', 'str'))
        ccdlist.add_row([self.expnum, self.ccdnum, self.band]) 

        for i,j in zip(survey.expnum[close], survey.band[close]):
            ccd = survey[i].checkInCCDFast(ra_arr, dec_arr, ccdsize = 0.149931)[1]
            if ccd != None:
                ccdlist.add_row([i,ccd,j])
        ccdlist.sort('EXPNUM')

        ccdlist['DETECTED'] = False
        ccdlist['DETECTED'][ccdlist['EXPNUM'] == self.expnum] = True

        self.exposures = tb.unique(ccdlist)

        if reduce_band:
            self.exposures = self.exposures[self.exposures['BAND'] == self.band]


        self.exposures.sort('DETECTED')

        if return_list:
            return self.exposures

    def findPixelCoords(self, expnum = None, ccdnum = None, pmc = DESMaps(), return_wcs = False, color = 0.61, ra = None, dec = None):
        '''
        Finds the pixel coordinates of the detection using pixmappy (data provided using the pmc argument)
        for a given expopsure/ccdnum pair. Can return the wcs for usage in other functions
        Color (g-i) is optional

        '''
        if expnum is None:
            expnum = self.expnum
        if ccdnum is None:
            ccdnum = self.ccdnum

        if ra is None:
            ra = self.ra
            dec = self.dec

        wcs = pmc.getDESWCS(expnum, int(ccdnum))

        x, y = wcs.toPix(np.array([ra]), np.array([dec]), c = color)

        if return_wcs:
            return x, y, wcs
        else:
            return x, y

    def constructImages(self, zeropoints, path, size = 30, background = False):
        '''
        Constructs the array of images in the format required for the linear algebra operations
        - zeropoints is a dictionary of ZP for each exposure/ccdnum, all exposures are brought to a common
        zeropoint = 30. 
        - path provides the location of all FITS for the exposures, the stamps should be
        names as {name}_EXPNUM.fits 
        - size is the size for the grid (size = 30 means 30x30 stamps)
        - background applies some background subtraction routines developed for the comet analysis

        '''

        #will be used for gain corrections later on
        self.zp = np.power(10, -(zeropoints[self.expnum][self.ccdnum] - 30)/2.5)


        m = []
        mask = []
        wgt = []
        bgflux = []
        for i in self.exposures:
            try:
                image = pf.open(f"{path}/{self.name}_{i['EXPNUM']}.fits")
            except OSError:
                if i['DETECTED']:
                    print('No stamp for the detection!')
                m.append(np.zeros(size*size))
                mask.append(np.ones(size*size))
                wgt.append(np.zeros(size*size))
                continue
            try:
                zero = np.power(10, -(zeropoints[i['EXPNUM']][i['CCDNUM']] - 30)/2.5)
            except:
                zero = -99
            if zero < 0:
                zero = 0
            im = image['SCI'].data * zero 
            bgarr = np.concatenate((im[0:size//10,0:size//10].flatten(), im[0:size,size//10:size].flatten(), im[size//10:size,0:size//10].flatten(), im[size//10:size,size//10:size].flatten()))
            bgarr = bgarr[bgarr != 0]
            if len(bgarr) == 0:
                med = 0
                bg = 0
            else:
                pc = np.percentile(bgarr, 84)
                med = np.median(bgarr)
                bgarr = bgarr[bgarr < pc]
                bg = np.median(bgarr)
            bgflux.append(bg)
            if background:
                im -= bg 
            m.append(im.flatten())
            mask.append(image['MSK'].data.flatten())
            w = zero**2/image['WGT'].data.flatten()
            w[image['WGT'].data.flatten() == 0] = 0 
            wgt.append(w)

        self.image = np.hstack(m)
        self.mask = np.hstack(mask)
        self.bgflux = bgflux

        self.wgt = np.hstack(wgt)

        self.invwgt = 1/self.wgt

        self.invwgt[self.mask > 0] = 0
        self.invwgt[self.wgt == 0] = 0


    def constructPSFs(self, ra_grid, dec_grid, pmc = DESMaps(), size = 30, shift_x = 0, shift_y = 0, path = '', sparse = False):
        '''
        Constructs the PIFF PSFs for the detections, requires an array of RA and Decs (ra_grid, dec_grid), a pixmappy instance (pmc),
        a stamp size, a potential offset in pixels for the center (shift_x,y), a path for the 
        PIFF files. 
        sparse turns on the sparse matrix solution (uses less memory and can be faster, but less stable)
        '''
        psf_matrix = []
        for i in self.exposures:
            try:
                x_cen, y_cen, wcs = self.findPixelCoords(i['EXPNUM'], int(i['CCDNUM']), pmc = pmc, return_wcs=True, color = self.color)
                psf = piff.PSF.read(f"{path}/{i['EXPNUM']}/{i['EXPNUM']}_{i['CCDNUM']}_piff.fits")
            except (OSError, ValueError):
                print(f"Missing {i['EXPNUM']} {i['CCDNUM']} psf")
                psf_matrix.append(sp.csr_matrix(np.zeros((size * size, len(ra_grid)))))
                continue
            psf_matrix.append(sp.csr_matrix(construct_psf_background(ra_grid, dec_grid, wcs, psf, x_cen, y_cen, size, flatten=True)))
        if sparse:
            print('PSF matrix')
            self.psf_matrix = sp.vstack(psf_matrix)
            del psf_matrix
        else:
            self.psf_matrix = np.vstack(psf_matrix)
        ## Last PSF is the one for the detected exposure
        self.x, self.y = self.findPixelCoords(pmc = pmc, color = self.color)
        self.source_psf = piff.PSF.read(f'{path}/{self.expnum}/{self.expnum}_{self.ccdnum}_piff.fits')
        self.psf_source = construct_psf_source(self.x + shift_x, self.y + shift_y, psf = self.source_psf, stampsize = size, x_center = self.x, y_center = self.y)

    def constructDesignMatrix(self, size, sparse = False, background = True):
        '''
        Constructs the design matrix for the solution. 
        size is the stamp size, sparse turns on the sparse solution
        background defines whether the background is being fit together with the image or not
        '''
        if not background:
            ones = np.ones((size*size,1))
        else:
            ones = np.zeros((size*size, 1))

        if sparse:
            print('Background')
            background = sp.block_diag(len(self.exposures) * [ones] )
        else:
            background = block_diag(*(len(self.exposures) * [ones]))

        psf_zeros = np.zeros((self.psf_matrix.shape[0]))

        psf_zeros[-size*size:] = self.psf_source

        if sparse:
            print('Design')
            self.design = sp.hstack([self.psf_matrix, background, np.array([psf_zeros]).T], dtype='float64')
        else:
            #self.design = sp.csc_matrix(self.design)
            self.design = np.column_stack([self.psf_matrix, background, psf_zeros])

    def solvePhotometry(self, res = True, err = True, sparse = False):
        '''
        Solves the system for the flux as well as background sources
        Solution is saved in det.X, the flux is the -1 entry in this array
        - res: defines if the residuals should be computed
        - err: defines if the errors should be computed (requires an expensive matrix inversion)
        - sparse: turns on sparse routines. Less stable, possibly incompatible with `err`
        '''
        if sparse:
            diag = sp.diags(np.sqrt(self.invwgt))
            print('Product')

            prod = diag.dot(self.design)
            print('Solving')
            self.X = sp.linalg.lsqr(prod, self.image*np.sqrt(self.invwgt))[0]
            print('Solved')
        else:
            self.X = lstsq(np.diag(np.sqrt(self.invwgt)) @ self.design, self.image*np.sqrt(self.invwgt))[0]

        self.flux = self.X[-1] 

        self.mag = -2.5*np.log10(self.flux) + 30

        if res:
            self.pred = self.design @ self.X 
            self.res = self.pred - self.image

        if err:
            inv_cov = self.design.T @ np.diag(self.invwgt) @ self.design
            try:
                self.cov = np.linalg.inv(inv_cov)
            except LinAlgError:
                self.cov = np.linalg.pinv(inv_cov)
                
            self.sigma_flux = np.sqrt(self.cov[-1,-1])
            self.sigma_mag = 2.5*np.sqrt(self.cov[-1,-1]/(self.flux**2))/np.log(10)

    def writeFits(self, filename):
        '''
        Saves the solution as a FITS image, similar to `write`
        '''
        newfits = pf.HDUList([pf.PrimaryHDU(), 
                        pf.ImageHDU(self.design, name='DESIGN'),
                        pf.ImageHDU(self.image, name='IMAGE'),
                        pf.ImageHDU(self.wgt, name='WGT'),
                        pf.ImageHDU(self.X, name='SOLUTION')])

        newfits[0].header['RA'] = self.ra 
        newfits[0].header['DEC'] = self.dec 
        newfits[0].header['EXPNUM'] = self.expnum
        newfits[0].header['CCDNUM'] = self.ccdnum 
        newfits[0].header['BAND'] = self.band 

        newfits[0].header['MAG'] = self.mag 
        newfits[0].header['MAG_ERR'] = self.sigma_mag
        newfits[0].header['FLUX'] = self.flux 
        newfits[0].header['FLUX_ERR'] = self.sigma_flux

        newfits.writeto(filename)


    def runPhotometry(self, se_path, piff_path, zp, survey, pmc = DESMaps(), n_grid = 20, size = 30, offset_x = 0, offset_y = 0, 
                      sparse = False, err = True, res = True, background = False):
        '''
        Convenience function that performs all operations required by the photometry
        - se_path: path for the SE postage stamps
        - piff_path: path for the PIFF files
        - zp: zeropoint dictionary
        - survey: `DESTNOSIM` list of exposures
        - pmc: pixmappy instance for astrometry
        - n_grid: grid size for point sources in the background (adds n_grid x n_grid sources)
        - size: stamp size
        - offset_x,y: offset in the x and y pixel coordinates
        - sparse: sparse routines
        - err: turns on error estimation
        - res: computes residuals
        - background: background estimation
        '''
        self.findAllExposures(survey)

        ra_grid, dec_grid = local_grid(self.ra, self.dec,0.35/3600, n_grid,)

        self.constructImages(zp, se_path, size = size, background = background)

        self.constructPSFs(ra_grid, dec_grid, pmc, size, offset_x, offset_y, piff_path, sparse = sparse)

        self.constructDesignMatrix(size, sparse, background = background)
        self.solvePhotometry(sparse = sparse, err = err, res = res)

    def photometryShotNoise(self, stampsize, gain_dict):
        '''
        Adds in shot noise estimates from a previous fit
        '''

        if self.flux > 0:
            ## fight gain
            gain_cut = gain(self.expnum, self.ccdnum, self.x, self.y, stampsize, gain_dict)
            gain_cut /= self.zp 

            sigma_photon = self.pred[-stampsize*stampsize:] / gain_cut.flatten()
            sigma_photon[sigma_photon < 0] = 0
            sigma_photon[np.isnan(sigma_photon)] = 0
            sigma_photon[np.isinf(sigma_photon)] = 0
            ## update weights
            self.wgt_shotnoise = np.copy(self.wgt) 

            self.wgt_shotnoise[-stampsize*stampsize:] += sigma_photon

            self.invwgt_shotnoise = 1/self.wgt_shotnoise
            self.invwgt_shotnoise[self.wgt_shotnoise == 0] = 0

            self.invwgt_shotnoise[self.wgt_shotnoise < 0] = 0

            self.invwgt_shotnoise[np.isnan(self.invwgt_shotnoise)] = 0
            self.invwgt_shotnoise[np.isinf(self.invwgt_shotnoise)] = 0

            #self.design[np.isnan(self.design)] = 0
            #self.design[np.isinf(self.design)] = 0

            #self.image[np.isnan(self.image)] = 0
            #self.image[np.isinf(self.image)] = 0

            
            ## redo photometry

            self.X_shotnoise = lstsq(np.diag(np.sqrt(self.invwgt_shotnoise)) @ self.design, self.image*np.sqrt(self.invwgt_shotnoise))[0]

            self.flux_shotnoise = self.X_shotnoise[-1]

            inv_cov = self.design.T @ np.diag(self.invwgt_shotnoise) @ self.design
            try:
                self.cov_shotnoise = np.linalg.inv(inv_cov)
            except LinAlgError:
                self.cov_shotnoise = np.linalg.pinv(inv_cov)
            self.sigma_flux_shotnoise = np.sqrt(self.cov_shotnoise[-1,-1])

        else:
            self.flux_shotnoise = self.flux
            self.sigma_flux_shotnoise = self.sigma_flux
            self.X_shotnoise = self.X
            self.cov_shotnoise = self.cov

        self.pred_shotnoise = self.design @ self.X_shotnoise

        self.mag_shotnoise = -2.5 * np.log10(self.flux_shotnoise) + 30
        self.sigma_mag_shotnoise = 2.5*self.sigma_flux_shotnoise/np.sqrt((self.flux_shotnoise**2))/np.log(10)

    def minimizeChisq(self, x_init, size=30, sparse = True, background = False, method='Powell'):
        from scipy.optimize import minimize

        self.solution = minimize(chi2_single, x_init, method =method, args = (self, sparse, size, background), options={'xtol' : 0.01})
        x_sol = self.solution.x 
        self.psf_source = construct_psf_source(self.x + x_sol[0], self.y + x_sol[1], self.source_psf, size, self.x, self.y)
        self.constructDesignMatrix(size, sparse, background)
        self.solvePhotometry(True, True, sparse)


class BinaryDetection(Detection):
    def constructPSFs(self, ra_grid, dec_grid, pmc = DESMaps(), size = 30, shift_x = 0, shift_y = 0, path = '', sparse = False, shift_x_binary = 0, shift_y_binary = 0):
        super().constructPSFs(ra_grid, dec_grid, pmc, size, shift_x, shift_y, path, sparse)
        self.psf_primary = self.psf_source
        self.psf_secondary = construct_psf_source(self.x + shift_x_binary, self.y + shift_y_binary, psf = self.source_psf, stampsize = size, x_center = self.x, y_center = self.y)

    def constructDesignMatrix(self, size, sparse = False, background = True):
        '''
        Constructs the design matrix for the solution. 
        size is the stamp size, sparse turns on the sparse solution
        background defines whether the background is being fit together with the image or not
        '''
        if not background:
            ones = np.ones((size*size,1))
        else:
            ones = np.zeros((size*size, 1))

        if sparse:
            print('Background')
            background = sp.block_diag(len(self.exposures) * [ones] )
        else:
            background = block_diag(*(len(self.exposures) * [ones]))

        psf_zeros_primary = np.zeros((self.psf_matrix.shape[0]))

        psf_zeros_primary[-size*size:] = self.psf_primary

        psf_zeros_secondary = np.zeros((self.psf_matrix.shape[0]))
        psf_zeros_secondary[-size*size:] = self.psf_secondary

        if sparse:
            print('Design')
            self.design = sp.hstack([self.psf_matrix, background, np.array([psf_zeros_primary]).T, np.array([psf_zeros_secondary]).T], dtype='float64')
        else:
            #self.design = sp.csc_matrix(self.design)
            self.design = np.column_stack([self.psf_matrix, background, psf_zeros_primary, psf_zeros_secondary])

    def solvePhotometry(self, res = True, err = True, sparse = False):
        super().solvePhotometry(res, err, sparse)

        self.flux_primary = self.X[-2]
        self.flux = self.X[-2]
        self.mag_primary = -2.5 * np.log10(self.flux_primary) + 30 
        if err:
            self.sigma_flux_primary = np.sqrt(self.cov[-2,-2])
            self.sigma_mag_primary = 2.5*np.sqrt(self.cov[-2,-2]/(self.flux_primary**2))/np.log(10)


        self.flux_secondary = self.X[-1]
        self.mag_secondary = -2.5 * np.log10(self.flux_secondary) + 30 
        if err:
            self.sigma_flux_secondary = np.sqrt(self.cov[-1,-1])
            self.sigma_mag_secondary = 2.5*np.sqrt(self.cov[-1,-1]/(self.flux_secondary**2))/np.log(10)

    def runPhotometry(self, se_path, piff_path, zp, survey, pmc = DESMaps(), n_grid = 20, size = 30, offset_x = 0, offset_y = 0, shift_x_binary = 0, shift_y_binary = 0, sparse = False, err = True, res = True, background = False):
        '''
        Convenience function that performs all operations required by the photometry
        - se_path: path for the SE postage stamps
        - piff_path: path for the PIFF files
        - zp: zeropoint dictionary
        - survey: `DESTNOSIM` list of exposures
        - pmc: pixmappy instance for astrometry
        - n_grid: grid size for point sources in the background (adds n_grid x n_grid sources)
        - size: stamp size
        - offset_x,y: offset in the x and y pixel coordinates
        - shift_x,y_binary: offset for the secondary point source
        - sparse: sparse routines
        - err: turns on error estimation
        - res: computes residuals
        - background: background estimation
        '''
        self.findAllExposures(survey)

        ra_grid, dec_grid = local_grid(self.ra, self.dec,0.35/3600, n_grid,)

        self.constructImages(zp, se_path, size = size, background = background)

        self.constructPSFs(ra_grid, dec_grid, pmc, size, offset_x, offset_y, piff_path, sparse, shift_x_binary, shift_y_binary)

        self.constructDesignMatrix(size, sparse, background = background)
        self.solvePhotometry(sparse = sparse, err = err, res = res)


    def photometryShotNoise(self, stampsize, gain_dict):
        super().photometryShotNoise(stampsize, gain_dict)
        self.flux_primary_shotnoise = self.X_shotnoise[-2]
        self.mag_primary_shotnoise = -2.5 * np.log10(self.flux_primary_shotnoise) + 30 
        self.sigma_flux_primary_shotnoise = np.sqrt(self.cov_shotnoise[-2,-2])
        self.sigma_mag_primary_shotnoise = 2.5*np.sqrt(self.cov_shotnoise[-2,-2]/(self.flux_primary_shotnoise**2))/np.log(10)


        self.flux_secondary_shotnoise = self.X_shotnoise[-1]
        self.mag_secondary_shotnoise = -2.5 * np.log10(self.flux_secondary_shotnoise) + 30 
        self.sigma_flux_secondary_shotnoise = np.sqrt(self.cov_shotnoise[-1,-1])
        self.sigma_mag_secondary_shotnoise = 2.5*np.sqrt(self.cov_shotnoise[-1,-1]/(self.flux_secondary_shotnoise**2))/np.log(10)

    def minimizeChisq(self, x_init, size=30, sparse = True, background = False, method='Powell'):
        from scipy.optimize import minimize

        self.solution = minimize(chi2_binary, x_init, method =method, args = (self, sparse, size, background), options={'xtol' : 0.001})
        x_sol = self.solution.x 
        self.psf_primary = construct_psf_source(self.x + x_sol[0], self.y + x_sol[2], self.source_psf, size, self.x, self.y)
        self.psf_secondary = construct_psf_source(self.x + x_sol[1], self.y + x_sol[3], self.source_psf, size, self.x, self.y)
        self.constructDesignMatrix(size, sparse, background)
        self.solvePhotometry(True, True, sparse)



def chi2_binary(x, detection, sparse = True, size = 30, background = False):
    x1, x2, y1, y2 = x 
    detection.psf_primary = construct_psf_source(detection.x + x1, detection.y + y1, detection.source_psf, size, detection.x, detection.y)
    detection.psf_secondary = construct_psf_source(detection.x  + x2, detection.y + y2, detection.source_psf, size, detection.x, detection.y)
    detection.constructDesignMatrix(size, sparse, background)
    detection.solvePhotometry(True, False, sparse)
    chisq = np.sum(detection.res * detection.res * detection.invwgt)

    return chisq 


def chi2_single(x, detection, size = 30, background = False):
    x1, y1 = x 
    detection.psf_source = construct_psf_source(detection.x + x1, detection.y + y1, detection.source_psf, size, detection.x, detection.y)
    detection.constructDesignMatrix(size, True, background)
    detection.solvePhotometry(True, False, True)
    chisq = np.sum(detection.res * detection.res * detection.invwgt)

    return chisq 



