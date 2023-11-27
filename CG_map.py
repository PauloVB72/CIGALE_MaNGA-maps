import pandas as pd
import marvin
import re
from marvin.tools import Cube , Image
from marvin import config
from astropy.io import ascii
from marvin.tools.quantities.map import Map
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from zmq import EVENT_CLOSE_FAILED
#import whan
#from whan import WHAN
from astropy import coordinates, units as u, wcs
from astropy.units import cds
from astropy.units import Quantity as q
from marvin.tools import Maps 
import os
import matplotlib.style as style 
import glob 
import time
#from progressbar import *   
import time
from astropy.cosmology import FLRW, LambdaCDM
from numpy import asarray
from numpy import loadtxt
from datetime import date
from numpy import ndarray
from os import path
from curses.ascii import isdigit
import configparser
from numpy import savetxt,loadtxt
import time
from matplotlib.pyplot import cm
from scipy import sparse, interpolate
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from marvin.tools.maps import Maps
import marvin.utils.plot.map as mapplot
from marvin.utils.general.images import showImage

from numpy import savetxt,loadtxt
import time
from matplotlib.pyplot import cm
from scipy import sparse, interpolate


#config.access = 'collab'
#config.login()
#config.setRelease('MPL-11')





class CG_Map:
       
    def __init__(self,path):
        
        path_drp = '/nsa_table.txt'
        
        self.tb_drp = ascii.read(path_drp)
        self.redshift = self.tb_drp['NSA_Z']
        self.sersic_phi = self.tb_drp['NSA_SERSIC_PHI']
        
        self.plateifu = path.split('/')[-1]
        self.TB = ascii.read(path+'/out/results.txt')
        self.spx = int(np.sqrt(self.TB['id'][-1]+1))
        
        try: 
            self.TB_2 = Table.read(path+'/out/observations.fits')
            
        except ImportError:
            
            ImportError( 'Error, empty data')

    def table(self):
        
        return self.TB
    
    
    def kpc(self):

        """
        kpc distances
        """
        #planck 2018 AOB
        num = np.where(self.tb_drp['PLATEIFU']==self.plateifu)[0][0]
        z = self.tb_drp['NSA_Z'][num]
        gal = LambdaCDM(H0 = 70, Om0 = 0.3111, Ode0 = 0.6889 , Tcmb0=2.7255, Neff=3.046, Ob0=None)
        d_a = gal.angular_diameter_distance(z)*u.kpc/u.Mpc 
        theta = 2.4241e-6
        l = 2*d_a * np.tan(theta/2)*10**3
        val = l.value
        
        return val
    
    
    def Col(self, column):
        """
        Column selection
        """
        try:
            if column in self.TB.columns:
                self.col = column
                return self.col
        except ImportError:
            raise ImportError("Error, this is not the correct column")

            
    def snr_mask(self,level):
             
        maps = Maps(self.plateifu)

        spx_snr = maps.spx_snr
        ha_low_snr = mapplot.mask_low_snr(spx_snr.value, spx_snr.ivar, snr_min=level)      

        return ha_low_snr
    
    def Val(self,column):
        
        col = self.Col(column)
        val = self.TB[str(col)]
        re_val = np.reshape(val,(self.spx,self.spx))
        
        return re_val
    
    def sr_den(self,column, snr):
        not_usecon = ['bayes.sfh.age_bq','bayes.sfh.r_sfr']
        
        
        if column == 'best.chi_square':
            
            val = self.Val(column)
            max_ = np.nanmax(val)
            val_data = val/max_
            
        elif column in not_usecon:
            
            val_ = self.Val(column)
            SNR = np.invert(self.snr_mask(snr))
            val_d = val_ * SNR
            val_real = val_ * SNR
            val_real[val_real==0.] = np.nan
            val_data = val_real
            
        else:

            val_ = self.Val(column)
            SNR = np.invert(self.snr_mask(snr))
            val_d = val_ * SNR
            val_real = val_ * SNR
            val_real[val_real==0.] = np.nan

            num = np.where(self.tb_drp['PLATEIFU']==self.plateifu)[0][0]
            phi = self.sersic_phi[num]
            area = (self.kpc()**2)*np.abs(np.cos(phi))
            val_data = val_real/area
        
        return val_data
    
    def extend(self,col,snr):
        
        val_s = self.sr_den(col,snr)
        max_v = np.nanmedian(val_s)+np.nanstd(val_s)*4
       # min_v = np.nanmin(val_s)
        return max_v
        
    def params(self,col,snr):
        
        dc_ = [self.sr_den(col,snr),self.extend(col,snr)]
        
        return dc_
    
class MC_plot:
    def __init__(self,path,column,snr,cmap,**kwargs):
    
        dc_ = CG_Map(path).params(column,snr)
        data = dc_[0]
        max_v = dc_[1]
        
        fig, ax = plt.subplots(1,sharex=False, sharey=False)
        #fig.subplots_adjust(top=1.3, bottom=0.2, left=0.02, right=0.95, hspace=0.25,
         #            wspace=0.35)         
        
  
        img = ax.imshow(data,origin='lower',vmax=max_v,cmap=cmap)
        ax.set_title(kwargs.get("title",None))
        cbar = plt.colorbar(img)
        cbar.set_label(kwargs.get("cb",None),fontsize = kwargs.get("fontsize_cb",10))
        cbar.outline.set_visible(False)
        ax.set_xlabel(kwargs.get("lx",None),fontsize = kwargs.get("fontsize_x",10))
        ax.set_ylabel(kwargs.get("ly",None),fontsize = kwargs.get("fontsize_y",10))
        cbar.ax.tick_params(labelsize=kwargs.get("labelsize",15))

        ax.tick_params(axis='x', labelsize=kwargs.get("labelsize_x",10))
        ax.tick_params(axis='y',labelsize=kwargs.get("labelsize_y",10))
        ax.tick_params(left = False ,
                 bottom = False)
        ax.patch.set(hatch='xxxx', edgecolor='white',facecolor='gray')
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft='on', labelbottom='on')
        
        color = 'white'
        ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines()
        for ticks in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
            ticks.set_color(color)
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor(color)
