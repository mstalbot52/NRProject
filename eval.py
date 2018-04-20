import numpy as np
from random import randrange, uniform
import random
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from scipy.optimize import minimize

class EvAl:
    '''Set up initial limits, data path, search parameters of the Evolution Algorythm + Nedler-Mead optimizations. Run search on a summed spectra (i.e. roughly flux) from a lens in MaNGA, and obtain best fit model parameters of a 2 component (ellipse + disk) model with reduced sersic core inflation.'''
    def __init__(self, limits = [(.5,14),(.5,14),(0,1),(0,90),(0,180),(1.5,4),(-.5,.5),(-.5,.5),(0,.3)], data_path = '/Users/michael/Downloads/manga-8606-6102-LOGCUBE.fits', mutate_occurance = .2, vars_to_mutate = 1, diff_sum_cut = .01, max_iterations = 100, population = 100, preserve = .2, slack = .15, segment_wave_range = [5500,8800], sample_radius_range = [2,7.5]):
        self.pars_array = {'pars':np.array([[uniform(lim[0], lim[1]) for i in range(population)] for lim in limits]).T, 'fits':np.ones(population)*9999}

        #Grab data               
        hdu = fits.open(data_path)
        self.flux = hdu['flux'].data
        self.wave = hdu['wave'].data
        hdu.close()
        
        #Set class variables
        self.sample_radius_range = sample_radius_range
        self.segment_wave_index = ((self.wave > np.min(segment_wave_range)) & (self.wave < np.max(segment_wave_range)))
        self.segment_flux = np.sum(self.flux[self.segment_wave_index,:,:],0)
        self.segment_wave_range = segment_wave_range                       
        self.population = population
        self.mutate_occurance = int(self.population * mutate_occurance)
        self.vars_to_mutate = vars_to_mutate
        self.diff_sum_cut = diff_sum_cut
        self.max_iterations = max_iterations
        self.limits = limits
        self.preserve = int(self.population * preserve)
        self.slack = int(1 + self.population * slack)

    def generate_model(self, index, final = False):
        '''Create model from a specific model'''
        self.model_index_temp = index
        model = self.target_shape(pars = self.pars_array['pars'][index], final = final)
        return model

    def measure_fit(self, model, pars):
        '''Find sum of flux-model difference'''
        HLRe, HLRd, contribution, inclination, rotation, psf, center_x, center_y, background = pars
        #Ignore the divergent sersic center
        shape = self.flux.shape
        x = np.arange(-shape[1]/2*.5, shape[1]/2*.5, .5)
        x_index = x.argmin()
        y = np.arange(-shape[2]/2*.5, shape[2]/2*.5, .5)
        y_index = y.argmin()
        x,y = np.meshgrid(x,y)
        r=np.sqrt(x**2 + y**2)
        measure_index = ((r<=self.sample_radius_range[1]) & (r>=self.sample_radius_range[0]))
        temp_res = np.abs(self.segment_flux[measure_index] - model[measure_index])
        return np.sum(temp_res)

    def set_grade(self):
        '''Grade the population model fitting'''
        for i in range(self.population):
            self.mark = i
            self.pars_array['fits'][i] = self.measure_fit(self.generate_model(i), self.pars_array['pars'][i])
        sort_index = np.array(self.pars_array['fits']).argsort()
        self.pars_array['fits'] = self.pars_array['fits'][sort_index]
        self.pars_array['pars'] = self.pars_array['pars'][sort_index]
        
    def run_generations(self):
        '''Driver to run all functions to perform Evolution Algorythm and Nedler-Mead combo'''
        shape = self.flux.shape
        x = np.arange(-shape[1]/2*.5, shape[1]/2*.5, .5)
        y = np.arange(-shape[2]/2*.5, shape[2]/2*.5, .5)
        x,y = np.meshgrid(x,y)
        
        self.gen = []
        self.find_nearest_minimum()
        for i in range(self.max_iterations):
            self.mile_marker = i
            print('run', i)
            self.set_grade()
            self.create_children()
            gen = self.pars_array
            self.gen.append(gen)
            
            if np.sum(np.abs(gen['pars'][:-self.slack,:] - np.mean(gen['pars'][:-self.slack,:], 0))) < self.diff_sum_cut:
                print('Success!!!!!', 'generation', gen['pars'][1], 'argmin', [int(np.argmin(self.flux)/100), np.argmin(self.flux)%100])
                break
            self.mutate_some_children()
            self.find_nearest_minimum()
            
            #Plot every 10 generation runs
            if self.mile_marker%10 == 0:
                temp_res = self.segment_flux - self.generate_model(0, final=False)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.contourf(x,y,temp_res, levels = np.linspace(np.min(temp_res), np.max(temp_res), 200))
                plt.colorbar(label = 'Flux-Model')
                plt.xlabel('X (Arcseconds from Galaxy Center)')
                plt.ylabel('Y (Arcseconds from Galaxy Center)')
                plt.title('Datacube Flux-Model')
                fig.savefig('/Users/michael/Documents/trial_marker_%s.png' %self.mile_marker)
                print('max', np.max(temp_res), 'min', np.min(temp_res))

        #Plot final difference image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        model = self.generate_model(0, final=False)
        temp_res = self.segment_flux - model
        plt.contourf(x,y,temp_res, levels = np.linspace(np.min(temp_res), np.max(temp_res), 200))
        plt.colorbar(label = 'Flux-Model')
        plt.xlabel('X (Arcseconds from Galaxy Center)')
        plt.ylabel('Y (Arcseconds from Galaxy Center)')
        plt.title('Segment Flux-Model')
        fig.savefig('/Users/michael/Documents/finish_%s.png' %self.mile_marker)

    def mutate_some_children(self):
        '''Change some random variables in some parameters of the next generation population'''
        for i in range(self.mutate_occurance):
            mutated_index = np.random.randint(self.preserve, self.population)
            for j in range(self.vars_to_mutate):
                var_index = np.random.randint(len(self.limits))
                lim = self.limits[var_index]
                self.pars_array['pars'][mutated_index, var_index] = np.random.uniform(lim[0],lim[1])

    def create_children(self):
        '''Create the next generation from the better fitting part of the population'''
        children = []
        length = len(self.pars_array['fits'])
        middle = int(len(self.limits)/2)
        for index in range(int(length/2)):
            children.append(np.append(self.pars_array['pars'][index,:middle], self.pars_array['pars'][index+1,middle:]))
            children.append(np.append(self.pars_array['pars'][index+1,:middle], self.pars_array['pars'][index,middle:]))
        children = {'pars':np.array(children), 'fits':np.ones(length)*9999}
        self.pars_array = children

    def plot_color_image(self):
        '''Plot a color image...more for fun and visual awareness than for the requirements of the search method'''
        wave = self.wave
        flux = self.flux
        
        #Create color bandwidths
        blue_index = ((wave > 4000) & (wave < 4500))
        green_index = ((wave > 5000) & (wave < 5600))
        red_index = ((wave > 6500) & (wave < 7200))
        
        #Create color fluxes
        blue = np.sum(flux[blue_index,:,:],0)
        green = np.sum(flux[green_index,:,:],0)
        red = np.sum(flux[red_index,:,:],0)

        #Create image data
        shape = flux.shape
        rgb = np.zeros((shape[1],shape[2],3))
        rgb[:,:,0] = red
        rgb[:,:,1] = green
        rgb[:,:,2] = blue
        rgb = np.log10(rgb)
        rgb[~np.isfinite(rgb)] = 0
        
        #Set colors to a color scale for the image
        rgb = np.array(rgb/np.max(rgb) * 255, dtype = np.uint8)
        
        #Plot image
        plt.imshow(rgb,norm=LogNorm(), origin='lower')
        plt.show(block=False)

    def target_shape(self, pars, scale_per_pixel = .5, final = False):
        '''Create 2 component model (ellipse + disk)'''
        #Set up parameters
        HLRe, HLRd, contribution, inclination, rotation, psf, center_x, center_y, background = pars
        #Create 2d radius data array and average flux
        shape = self.flux.shape
        temp_flux = np.sum(self.flux,0) if final else self.segment_flux
        
        I0 = np.mean(temp_flux[int(shape[1]/2)-20:int(shape[1]/2)+20, int(shape[2]/2)-20:int(shape[2]/2)+20])
        x = np.arange(-shape[1]/2*.5 - center_x, shape[1]/2*.5 - center_x, .5)
        y = np.arange(-shape[2]/2*.5 - center_y, shape[2]/2*.5 - center_y, .5)
        x,y = np.meshgrid(x,y)
        rotation = np.deg2rad(rotation)
        x2 = x*np.cos(rotation)+y*np.sin(rotation)
        y2 = -x*np.sin(rotation)+y*np.cos(rotation)
        r_ellipse = np.sqrt(x2**2+y2**2)
        r_disk = np.sqrt(x2**2+(y2/np.cos(inclination))**2)
        
        #Create 2d model intensity per radius
        I_ellipse = I0*contribution*np.exp(-7.67*((r_ellipse/HLRe)**(1/4)-1))
        I_disk = I0*(1-contribution)*np.exp(-r_disk/HLRd)
        I_background = I0*background
        I_total = I_ellipse + I_disk + I_background

        #PSF smoothing of 2d intensity if psf > .01
        kernel = Gaussian2DKernel(stddev=psf)
        try: I_total = convolve(I_total, kernel)
        except: pass
        I_total = I_total/np.max(I_total) * np.max(temp_flux[int(shape[1]/2)-6:int(shape[1]/2)+6, int(shape[2]/2)-6:int(shape[2]/2)+6])
        return I_total
    
    def objective_function(self, x):
        '''Create a y-f(x) for the Nelder-Meat algorythm'''
        return self.measure_fit(self.target_shape(x),x)        

    def run_NM(self, pars):
        '''Use Nelder-Mead to find nearest minimum'''
        opt = minimize(self.objective_function, pars, method='Nelder-Mead')
        return opt['x']

    def find_nearest_minimum(self):
        '''Find batch process of minimums'''
        for index, pars in enumerate(self.pars_array['pars']): self.pars_array['pars'][index] = self.run_NM(pars)
