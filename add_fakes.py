import pandas as pd
import numpy as np
import glob
import os
from astropy.wcs import WCS
from astropy.io import fits
from datastats.calculate_FWHM import calculate_FWHM
from astropy.modeling.functional_models import Moffat2D
import argparse
from models.afterglows import generate_afterglows

def add_star(data,X,Y,flux,FWHM,alpha=4.765):

    res = 5

    gamma=FWHM/(2.*np.sqrt(2**(1./alpha)-1))
    psf = Moffat2D(amplitude=flux,x_0=X,y_0=Y,alpha=alpha,gamma=gamma)

    xarr = np.linspace(-24.5,26.5,51*res+1)+int(np.round(X))
    yarr = np.linspace(-24.5,26.5,51*res+1)+int(np.round(Y))
    zarr = np.array([psf(xarr,y) for y in yarr])

    star = np.array([[np.trapz([
        np.trapz(z_x,xarr[i*res:i*res+res+1]) 
        for z_x in
        zarr[i*res:i*res+res+1,j*res:j*res+res+1]],yarr[j*res:j*res+res+1])
             for i in range(0,51)] for j in range(0,51)])
    for i in range(51):
        i_index = i-25+int(np.round(X))
        if (i_index > 0) & (i_index < len(data[0]) - 1):
            for j in range(51):
                j_index = j-25+int(np.round(Y))
                if (j_index > 0) & (j_index < len(data) - 1):
                    data[j_index,i_index] += star[i,j]

    return data

def distribute_events(rng,models,gal_coords,fitsfile):
    ra_max = np.max([fitsfiles[0][0].header['COR1RA1'],
                     fitsfiles[0][0].header['COR2RA1'],
                     fitsfiles[0][0].header['COR3RA1'],
                     fitsfiles[0][0].header['COR4RA1']])
    ra_min = np.min([fitsfiles[0][0].header['COR1RA1'],
                     fitsfiles[0][0].header['COR2RA1'],
                     fitsfiles[0][0].header['COR3RA1'],
                     fitsfiles[0][0].header['COR4RA1']])
    
    dec_max = np.max([fitsfiles[0][0].header['COR1DEC1'],
                      fitsfiles[0][0].header['COR2DEC1'],
                      fitsfiles[0][0].header['COR3DEC1'],
                      fitsfiles[0][0].header['COR4DEC1']])
    dec_min = np.min([fitsfiles[0][0].header['COR1DEC1'],
                      fitsfiles[0][0].header['COR2DEC1'],
                      fitsfiles[0][0].header['COR3DEC1'],
                      fitsfiles[0][0].header['COR4DEC1']])
    
    coords = np.array([rng.uniform(ra_min,ra_max,len(models)),
                       rng.uniform(dec_min,dec_max,len(models))])
    models = models.assign(ra = coords[0],dec = coords[1])
    
    n_gal_events = int(np.floor(len(models)/2))
    if n_gal_events > len(gal_coords):
        n_gal_events = len(gal_coords)

    indices = rng.choice(np.array(models.index),n_gal_events,replace=False)
    models = models.assign(gal = False)
    for i in range(n_gal_events):
        models.iloc[indices[i],-3] = gal_coords['ra'][i]
        models.iloc[indices[i],-2] = gal_coords['dec'][i]
        models.iloc[indices[i],-1] = True 
    return models

def add_fakes(models,light_curves,names,fitsfiles,cals):
    for i in range(len(fitsfiles)):
        if os.path.exists(names[i].replace('.fits','_fakes.fits')):
            continue
  
        wcs = WCS(fitsfiles[i][0].header)
        data = fitsfiles[i][0].data

        fwhm=fitsfiles[i][0].header['FWHM']/0.263
        
        Xs,Ys = wcs.wcs_world2pix(models['ra'],models['dec'],1)
        print(names[i],cals['zeropoint'][cals['file'] == names[i]])
        cal = float(cals['zeropoint'][cals['file'] == names[i]].iloc[0])
        mags = [float(light_curves[j]['gmag'][
            light_curves[j]['names'] == names[i]].iloc[0]) 
            for j in range(len(models))]
        mags = np.array([np.nan if mag > 24.5 else mag for mag in mags])
        fluxes = 10**(-(mags + cal)/2.5)
        
        for X,Y,flux in zip(Xs,Ys,fluxes):
            if np.isnan(flux):
                continue
            else:
                data = add_star(data,X,Y,flux,fwhm)

        fitsfiles[i][0].data = data
        fitsfiles[i].writeto(names[i].replace('.fits','_fakes.fits'),
                             overwrite=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='correct photometry')
    parser.add_argument('-d', '--datadir',
                        type=str,
                        default='data/',
                        nargs=1,
                        help='Path to data where fakes are to be added.')
    parser.add_argument('-f', '--field_run',
                        type=str,
                        nargs=1,
                        help='The field run that fakes are to be added to \
                            with format <field>_<mm>_<yyyy>.')
    parser.add_argument('-c', '--ccd',
                        type=str,
                        default='all',
                        help='The CCD that fakes are to be added to.')
    parser.add_argument('-n', '--nevents',
                        type=int,
                        default=1000,
                        nargs=1,
                        help='Number of events to force to occur during \
                            observations.')
    parser.add_argument('-s', '--shorts',
                        type=int,
                        default=0,
                        nargs=1,
                        help='Number of events to force to occur during \
                            observations.')
    parser.add_argument('-m', '--model',
                        type=str,
                        default='afterglows',
                        nargs=1,
                        help='Model you want to use for fakes.')
    args = parser.parse_args()
 
    nevents = args.nevents[0]
    shorts = args.shorts[0]
    field_run = args.field_run[0]
    ccd = args.ccd

    tempdir = 'temp_' + ccd + '/'
    if os.path.exists(tempdir) == False:   
        os.makedirs(tempdir)
    os.chdir(tempdir)

    rng = np.random.default_rng()

    cals = pd.read_csv(args.datadir[0] + field_run + '/cals/cals_' + 
                       ccd + '.csv')
    
    if os.path.exists(args.datadir[0] + field_run
                      + '/light_curves/' + ccd + '/') == False:
        os.makedirs(args.datadir[0] + field_run 
                    + '/light_curves/' + ccd + '/')

    print("Reading in fits files ...")
    fitsnames = glob.glob(args.datadir[0] + field_run +
                      '/*/c4d_*_ooi_g_v1/c4d_*_ooi_g_v1_' 
                      + ccd + '.fits')
    fitsnames = np.sort(np.array(fitsnames))
    fitsfiles = [fits.open(file) for file in fitsnames]
    print("Done!")
    MJDs = np.array([file[0].header['MJD-OBS'] for file in fitsfiles])
    
    gal_coords = pd.read_csv(args.datadir[0] + field_run + 
                             '/gals/gals_' + ccd + '.csv')
    
    if os.path.exists(args.datadir[0] + field_run + '/models/') == False:
        os.makedirs(args.datadir[0] + field_run + '/models/')
    if os.path.exists(args.datadir[0] + field_run + 
                      '/models/models_' + ccd + '.csv'):
        models = pd.read_csv(args.datadir[0] + field_run + 
                             '/models/models_' + ccd + '.csv')
    else:
        print("Generating models ...")
        if args.model[0] == 'afterglows':
            models = generate_afterglows(rng,nevents,fitsnames,MJDs,
                                         args.datadir[0] + field_run + 
                                         '/light_curves/' + ccd + '/',
                                         shorts=shorts)
        models = models.assign(lc_name=models.index.astype(str) 
                                       + '.csv')
        models = models.reset_index(drop=True)

        models = distribute_events(rng,models,gal_coords,fitsfiles[0])
        models.to_csv(args.datadir[0] + field_run + 
                          '/models/models_' + ccd + '.csv',index=False)
        print("Done!")
    
    light_curves = [pd.read_csv(args.datadir[0] + field_run + 
                                '/light_curves/' + ccd + '/' + lc_name) 
                    for lc_name in models.lc_name]
    
    print("Adding models to images ...")
    add_fakes(models,light_curves,fitsnames,fitsfiles,cals)
    os.chdir('../')
    os.system('rm -r ' + tempdir)
    print("All Done!")
