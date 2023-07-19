import pandas as pd
import numpy as np
import astropy.units as u
import glob
import os
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.special import erf
from functools import partial
from astropy.io import fits
import afterglowpy as grb
from astropy.cosmology import Planck18 as cosmo
import dask
import dask.dataframe as dd
from prep_phot_funcs.extract_sources import extract_sources
from phot_funcs.correct_photometry import photom_correct
from datastats.calculate_FWHM import calculate_FWHM
from astropy.modeling.functional_models import Moffat2D
from scipy.integrate import simps

def add_star(data,X,Y,flux,FWHM,alpha=4.765):
   
    res = 5

    gamma=FWHM/(2.*np.sqrt(2**(1./alpha)-1))
    psf = Moffat2D(amplitude=flux,x_0=X,y_0=Y,alpha=alpha,gamma=gamma)

    xarr = np.linspace(-24.5,26.5,51*res+1)+int(np.round(X))
    yarr = np.linspace(-24.5,26.5,51*res+1)+int(np.round(Y))
    zarr = np.array([psf(xarr,y) for y in yarr])

    star = np.array([[np.trapz([
        np.trapz(z_x,xarr[i*res:i*res+res+1]) for z_x in zarr[i*res:i*res+res+1,j*res:j*res+res+1]],yarr[j*res:j*res+res+1])
             for i in range(0,51)] for j in range(0,51)])
    for i in range(51):
        i_index = i-25+int(np.round(X))
        if (i_index > 0) & (i_index < len(data[0]) - 1):
            for j in range(51):
                j_index = j-25+int(np.round(Y))
                if (j_index > 0) & (j_index < len(data) - 1):
                    data[j_index,i_index] += star[i,j]

    return data

#LOOP THROUGH ALL FILES

class phot_args:
    catalog = None
    coords = None
    extension = '.cat'
    files = None
    filters = ['g']
    gaia = ''
    linear_regression=False
    mag_cut=None
    overwrite=False
    plots=False

field_run = 'FRB190711_09_2022_22'
files = glob.glob('data/' + field_run + '/*/*/*ext22.fits')

phot_args.catalog = ['/fred/oz100/jfreeburn/light_curve_test/09_2022/FRB190711/FRB190711_g_1_SM_phot.cat']
phot_args.mag_cut = '15.5,18.5'
phot_args.variables = ['/fred/oz100/jfreeburn/light_curve_test/09_2022/FRB190711/variables.csv']

cals = []
cal_std = []
sextractor = 'sex'

for file in files:
    print(file)
    
    fwhm=np.abs(calculate_FWHM(
                ['/fred/oz100/jfreeburn/fakes/' + file],
                sextractorloc=sextractor,
                verbose=False,
                quietmode=True)[0])

    Xs = np.array([1150]*5)
    Ys = np.linspace(2600,3300,5)
    fluxes = np.geomspace(500,10000,5)

    hdul = fits.open(file)
    data = hdul[0].data
    wcs = WCS(hdul[0].header)

    for X,Y,flux in zip(Xs,Ys,fluxes):
        data = add_star(data,X,Y,flux,fwhm)

    hdul[0].data = data
    if os.path.exists('data/' + field_run + '/cals/')==False:
        os.makedirs('data/' + field_run + '/cals/')
    hdul[0].writeto('data/' + field_run + '/cals/' + file.split('/')[-1],
                    overwrite=True)
    
    extract_sources('data/' + field_run + '/cals/' + file.split('/')[-1], 
                    'data/' + field_run + '/cals/',sextractor)
    phot_args.files = [('data/' + field_run + '/cals/' + file.split('/')[-1]).replace(
        '.fits','.cat')]
    photom_correct(phot_args)

    df = pd.read_csv(('data/' + field_run + '/cals/' + file.split('/')[-1]).replace(
         '.fits','_corr.csv'))
    
    mags = []
    for X,Y in zip(Xs,Ys):
        distances = [np.sqrt((df_X-X)**2 + (df_Y-Y)**2)
                     for df_X,df_Y in zip(np.array(df['X_IMAGE']),np.array(df['Y_IMAGE']))]
        mindex = np.argmin(distances)
        mags.append(df.iloc[mindex]['MAG_AUTO'])
    mags = np.sort(mags)[::-1]
    calibration = np.average(-2.5*np.log10(fluxes) - mags)
    cals.append(calibration)
    stdev = np.std(-2.5*np.log10(fluxes) - mags)
    cal_std.append(stdev)
cal_df = pd.DataFrame(np.transpose(np.array([files,cals,cal_std])),columns=['file','zeropoint'])
cal_df.to_csv('data/' + field_run + '/cals.csv',index=False)
os.system('rm -r ' + 'data/' + field_run + '/cals/')
