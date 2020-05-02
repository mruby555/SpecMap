import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import constants


class SpecMap:
    def __init__(self,center, wave_range, fits_file, PolyDegree = 1, spectralAxis = 2):
        """
        Generate velocity, dispersion, and flux maps from a fits datacube given center wavelength and range.
	    center: wavelength of emission line in observed frame
	    wave_range: range around emission line to fit
	    PolyDegree: degree of polynomial used to fit continuum
	    spectralAxis: used to find correct CDELT, CRVAL, and CRPIX values in header (0, 1, or 2) 
        """
        self.hdr = fits_file.header
        self.center = center
        self.polyDeg = PolyDegree
        
        #build spectral cube
        cube = fits_file.data
        shape = cube.shape
        for i,ax in enumerate(shape):
            if (ax > 1000):
                spec_ax = i


        if (spectralAxis == 0):
            self.delta = self.hdr['CDELT1']         #spectral resolution
            centerVal = self.hdr['CRVAL1']          #wavelength corresponding to center pix
            try:
                refPix = self.hdr['CRPIX1']             #center pix
            except:
                print("CRPIX1 not found!")
                refPix = int(input("Enter reference pixel value: "))
            try:
                self.specUnit = self.hdr['CUNIT1']
            except:
                print("CUNIT1 not found! Setting specUnit to 'um'")
                self.specUnit = 'um'
        elif(spectralAxis == 1):
            self.delta = self.hdr['CDELT2']
            centerVal = self.hdr['CRVAL2']
            try:
                print("CRPIX2 not found!")
                refPix = int(input("Enter reference pixel value: "))
            except:
                refPix = 0
            try:
                self.specUnit = self.hdr['CUNIT2']
            except:
                print("CUNIT2 not found! Setting specUnit to 'um'")
                self.specUnit = 'um'
        elif(spectralAxis == 2):
            self.delta = self.hdr['CDELT3']
            centerVal = self.hdr['CRVAL3'] 
            try:
                refPix = self.hdr['CRPIX3']
            except:
                print("CRPIX3 not found!")
                refPix = int(input("Enter reference pixel value: "))
            try:
                self.specUnit = self.hdr['CUNIT3']
            except:
                print("CUNIT3 not found! Setting specUnit to 'um'")
                self.specUnit = 'um'
        else:
            raise ValueError("spectralAxis value not valid...")
        #setup for calculating velocities using pixels instead of wavelength
        self.pix_range = wave_range / self.delta
        self.pix_center = (center - centerVal) / self.delta + refPix

        #set range for generating spectral slab
        lo = int(self.pix_center - self.pix_range / 2)
        hi = int(self.pix_center + self.pix_range / 2)

        #reshape data to place spectral axis in dim 3
        cube = np.moveaxis(cube,spec_ax,-1)

        #generate spectral slab
        data = cube[:,:,lo:hi]
        self.shape0 = data.shape[0]
        self.shape1 = data.shape[1]
        self.shape2 = data.shape[2]
        print("Smoothing...")
        self.data = self.__smooth(data)

        #set up fitting parameters and initialize LevMarLSQFitter
        #this avoids rebuilding model each iteration
        
        #build model, line (or polynomial) + gaussian
        gauss = models.Gaussian1D(mean=self.pix_center)
        gauss.amplitude.min = 0     #prevents fitting of absorption lines
        gauss.stddev.min = 1        #limits fwhm to about 2.3 pixels to prevent fitting noise
        gauss.stddev.max = 50        #prevents fitting gaussian to continuum, max FWHM = 115 pixels
        continuum = models.Polynomial1D(PolyDegree)
        self.__m_init = gauss + continuum
        self.__fit = fitting.LevMarLSQFitter()

        #create 1D list of spectrums
        spectrums = [self.data[row,column,:] for row in range(self.shape0) for column in range(self.shape1)]
        
        #create np arrays to hold velocities, fwhm, flux, and continuum
        self.velocities = np.zeros((self.shape0,self.shape1))
        self.fwhm = np.zeros((self.shape0,self.shape1))
        self.flux = np.zeros((self.shape0,self.shape1))
        self.continuum = np.zeros((self.shape0,self.shape1))

        #fit model to each spectrum and calculate velocities, fwhm, and flux
        for i,spec in enumerate(spectrums):
            self.__fit_spec(i,spec)

        if self.specUnit == 'nm':
            print("Adjusting flux values for nanometers...")
            self.flux /= 1000

        print("Done!")


    def __fit_spec(self,i,spec):
        row = i//self.shape1
        col = i%self.shape1
        x = np.linspace(self.pix_center-self.pix_range/2,self.pix_center+self.pix_range/2,self.shape2)
        m = self.__fit(self.__m_init,x,spec)  #fit model

        #evaluate goodness of fit and throw out bad fits
        if(self.__fit.fit_info['ierr'] not in [1,2,3,4]) or (type(self.__fit.fit_info['cov_x']) == type(None) or (m.amplitude_0.value < 1e-12)):
            self.velocities[row,col] = np.nan
            self.fwhm[row,col] = np.nan
            self.flux[row,col] = np.nan
        else:
            #convert mean pixel shift to velocity in km/s
            self.velocities[row,col] = self.__pix_to_vel(m.mean_0.value - self.pix_center)
        
            #calculate fwhm from sigma: FWHM = 2*sqrt(2*ln(2))*sigma after converting sigma to km/s
            self.fwhm[row,col] = self.__pix_to_vel(m.stddev_0.value)*2*np.sqrt(2*np.log(2))
        
            #calculate flux of emission line from flux = gaussian height * sigma(in wavelength) * sqrt(2*Pi)
            self.flux[row,col] = m.amplitude_0.value * m.stddev_0.value * self.delta * np.sqrt(2*np.pi)

        #store continuum values at emission line wavelength
        #continuum is measured flux at center - model(center) + modeled continuum(center)
        self.continuum[row,col] = spec[int(self.pix_range//2)] - m(self.pix_center)
        for deg in range(self.polyDeg):
            self.continuum[row,col] += (self.pix_center**deg)*getattr(m,m.param_names[deg+3])
        
        #print status
        if col == 0:
            print('Row: ', row)

        #uncomment to view subset of fits for verification
        """
        if col == 50:
            plt.plot(x, m(x))
            plt.plot(self.data[row,col,:]
            plt.show()
        """
    
    def __pix_to_vel(self,pix):
        """
        convert pixel units to velocity in km/s
        """
        return pix*self.delta*constants.c.value/self.center/1000

    def __smooth(self,data):
        """
        Smooth data by averaging each data point in 3X3 square.
        Avoid edges
        """
        newData = np.full_like(data,np.nan,dtype=np.double)
        for row in range(self.shape0 - 2):
            for column in range(self.shape1 - 2):
                for pix in range(self.shape2):
                    newData[row + 1,column + 1,pix] = np.average([data[row + srow - 1,column + scol - 1,pix] for srow in range(3) for scol in range(3)])
        return newData

    def show_velocity(self,range_min = -150,range_max= 150, colormap = 'jet'):
        fig, axs = plt.subplots(1,1)
        c = axs.imshow(self.velocities, cmap=colormap, vmin = range_min, vmax=range_max, interpolation='nearest',origin='lower')
        fig.colorbar(c, ax=axs)
        plt.show()

    def show_fwhm(self,range_min = 0,range_max = 250, colormap = 'jet'):
        fig, axs = plt.subplots(1,1)
        c = axs.imshow(self.fwhm, cmap=colormap, vmin = range_min, vmax=range_max, interpolation='nearest',origin='lower')
        fig.colorbar(c, ax=axs)
        plt.show()

    def show_flux(self,range_min = 0,range_max = 1e-4,colormap = 'viridis'):
        fig, axs = plt.subplots(1,1)
        c = axs.imshow(self.flux, cmap=colormap, vmin = range_min, vmax=range_max, interpolation='nearest',origin='lower')
        fig.colorbar(c, ax=axs)
        plt.show()

    def show_continuum(self,range_min=0,range_max = 1, colormap = 'viridis'):
        fig, axs = plt.subplots(1,1)
        c = axs.imshow(self.continuum, cmap=colormap, vmin = range_min, vmax=range_max, interpolation='nearest',origin='lower')
        fig.colorbar(c, ax=axs)
        plt.show()
        
    def save_fits(self):
        primary_hdu = fits.PrimaryHDU(header=self.hdr)
        velMap_hdu = fits.ImageHDU(self.velocities,header = self.hdr)
        disMap_hdu = fits.ImageHDU(self.fwhm,header = self.hdr)
        flxMap_hdu = fits.ImageHDU(self.flux,header = self.hdr)
        conMap_hdu = fits.ImageHDU(self.continuum,header=self.hdr)

        hdul = fits.HDUList([primary_hdu,velMap_hdu,disMap_hdu,flxMap_hdu,conMap_hdu])
        f_name = input("Enter filename, including .fits extention: ")
        hdul.writeto(f_name,output_verify='warn')


if __name__ == '__main__':
    file_name = '/home/m_ruby/OneDrive/Astro Research/Datacubes/NGC 7469/ngc7469_k035_pa132_lgs.fits'

    
    #preprocessing
    with fits.open(file_name, ignore_blank=True) as fits_file:
        fits_file.verify(option='silentfix')
        fits_file.info()
        maps = SpecMap(2.15635, 0.02, fits_file[0], spectralAxis=0)
        
    maps.show_velocity()
    maps.show_fwhm()
    maps.show_flux()
    maps.save_fits()
