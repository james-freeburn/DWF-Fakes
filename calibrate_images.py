import pandas as pd
import numpy as np
import glob
import os
import argparse
import subprocess
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS 
from astropy.io import fits
from astropy.stats import sigma_clip
from phot_funcs.correct_photometry import photom_correct
from astropy.modeling.functional_models import Moffat2D

def cross_match(obs_cat, viz_cat):
    radius_threshold = 2*u.arcsec

    coords_viz = SkyCoord(ra=viz_cat.RA, dec=viz_cat.DEC, unit='deg')
    coords_obs = SkyCoord(ra=obs_cat.RA, dec=obs_cat.DEC, unit='deg')

    idx, d2d, d3d = coords_viz.match_to_catalog_3d(coords_obs)

    sep_constraint = d2d <= radius_threshold

    # Get entries in cat_ref with a match
    viz_matched = viz_cat[sep_constraint]

    # Get matched entries in cat_sci
    obs_matched = obs_cat.iloc[idx[sep_constraint]]

    # re-index to match two dfs
    viz_matched = viz_matched.reset_index(drop=True)
    obs_matched = obs_matched.reset_index(drop=True)

    return obs_matched, viz_matched

def read_cat(file):
    with open(file,'r') as f:
        string = f.read()
        sexcolumns = [header.split()[1] for header in np.append(string.split('#')[1:-1],string.split('#')[-1].split('\n')[0])]
        f.close()
                                     
    cat = np.genfromtxt(file, unpack=True, filling_values=np.nan,
                        invalid_raise=False)
    cat = pd.DataFrame(data=np.transpose(cat), columns=sexcolumns)
    return cat

def query_gaia(ra,dec):
    cat = 'I/355/gaiadr3'
    # Create a SkyCoord object
    coords = SkyCoord(ra=ra * u.deg,
                      dec=dec * u.deg,
                      frame='icrs')
    # Select only those columns with relevant info for Scamp
    columns_select = ['RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS',
                        'Source', 'Gmag', 'e_Gmag', 'Plx']
    # Vizier object
    v = Vizier(columns=columns_select, row_limit=-1)
    # Query Vizier
    t_result = v.query_region(coords, width=18.*u.arcmin,
                                height=9.*u.arcmin,
                                catalog=cat)
    # Check if the sources were found
    if len(t_result[cat]) == 0:
        return None
    else:
        return t_result[cat].to_pandas()

def extract_sources(fitsfile, config_dir):
    catfilename = fitsfile.replace('.fits', '.cat')
    command =   f"sex -c {config_dir}/default.config "\
                f"-CATALOG_NAME {catfilename} "\
                f"-PARAMETERS_NAME {config_dir}/default.param -FILTER_NAME {config_dir}/default.conv "\
                f"{fitsfile}"
    subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

def get_fwhm(file,ccd,config):
    with fits.open(file) as hdul:
        ext = 0 if len(hdul) == 1 else 'S4'
        hdu = hdul[ext]

    extract_sources(file, args.config)

    cat = read_cat(file.replace('.fits', '.cat'))
    os.remove(file.replace('.fits', '.cat'))
    cat.rename(columns={'X_WORLD':'RA', 'Y_WORLD':'DEC'}, inplace=True)
    cat = cat[(cat['MAGERR_AUTO'] < 0.1) & (cat['ELLIPTICITY'] < 0.2)].reset_index(drop=True)

    if os.path.exists(f"gaia_{ccd}.csv"):
        gaia = pd.read_csv(f"gaia_{ccd}.csv")
    else:
        gaia = query_gaia(hdu.header['CRVAL1'], hdu.header['CRVAL2'])
        gaia.rename(columns={'RA_ICRS':'RA','DE_ICRS':'DEC'}, inplace=True)
        gaia = gaia[~np.isnan(gaia['Plx'])].reset_index(drop=True)
        gaia.to_csv(f"gaia_{ccd}.csv",index=False)

    cat_matched,gaia_matched = cross_match(cat, gaia)

    FWHMs = np.array(cat_matched.FWHM_IMAGE)*0.263
    FWHMs = FWHMs[~sigma_clip(FWHMs, sigma=2.0).mask]

    with fits.open(file, mode='update') as hdul:
        hdr = hdul[0].header
        hdr['FWHM'] = (np.median(FWHMs), 'Median FWHM of sources in arcsec')
        hdul.flush()

    return np.median(FWHMs)

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
                        help='The CCD that fakes are to be added to.')
    parser.add_argument('-m', '--mag_cut',
                        type=str,
                        default='0,40',
                        help='Range of magnitudes to correct photometry from')
    parser.add_argument('-p', '--catalog',
                        type=str,
                        nargs=1,
                        help='Catalog to perform photometry with.')
    parser.add_argument('-v', '--variables',
                        type=str,
                        default='',
                        nargs=1,
                        help='Variable catalog to remove variable stars.')
    parser.add_argument('-g', '--gaia',
                        type=str,
                        default='',
                        nargs=1,
                        help='Gaia catalog to perform parallax cuts.')
    parser.add_argument('-s', '--config',
                        type=str,
                        default='config/',
                        help='Path to SExtractor config directory.')
    args = parser.parse_args()
 
    field_run = args.field_run[0]
    ccd = args.ccd
    phot_args.catalog = args.catalog
    phot_args.mag_cut = args.mag_cut
    phot_args.gaia = args.gaia
    phot_args.variables = args.variables
    phot_args.error_correction = False
    files = glob.glob(args.datadir[0] + field_run  + '/*/*/*' 
                      + ccd + '.fits')
    cals = []
    cal_std = []
    sextractor = 'sex'
   
    Xs = np.array([1150]*5)
    Ys = np.linspace(2600,3300,5)
    fluxes = np.geomspace(500,10000,5)
    
    for file in files:
        if os.path.exists((args.datadir[0] + field_run + '/cals_' + ccd
                           + 'temp/'
                           + file.split('/')[-1]).replace(
                               '.fits','_corr.csv')):

            df = pd.read_csv((args.datadir[0] + field_run + '/cals_' + ccd
                          + 'temp/'
                          + file.split('/')[-1]).replace('.fits','_corr.csv'))

            mags = []
            for X,Y in zip(Xs,Ys):
                distances = [np.sqrt((df_X-X)**2 + (df_Y-Y)**2)
                         for df_X,df_Y in
                         zip(np.array(df['X_IMAGE']),np.array(df['Y_IMAGE']))]
                mindex = np.argmin(distances)
                mags.append(df.iloc[mindex]['MAG'])
            mags = np.sort(mags)[::-1]
            calibration = np.median(-2.5*np.log10(fluxes) - mags)
            cals.append(calibration)
            stdev = np.std(-2.5*np.log10(fluxes) - mags)
            cal_std.append(stdev)
            continue
        fwhm=get_fwhm(file,ccd,args.config)/0.263
    
        hdul = fits.open(file)
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    
        for X,Y,flux in zip(Xs,Ys,fluxes):
            data = add_star(data,X,Y,flux,fwhm)
    
        hdul[0].data = data
        if os.path.exists(args.datadir[0] + field_run + '/cals_' + ccd 
                          + 'temp/')==False:
            os.makedirs(args.datadir[0] + field_run + '/cals_' + ccd 
                        + 'temp/')
        hdul[0].writeto(args.datadir[0] + field_run + '/cals_' + ccd 
                        + 'temp/' + 
                        file.split('/')[-1],overwrite=True)
        
        extract_sources(args.datadir[0] + field_run + '/cals_' + ccd 
                        + 'temp/' + 
                        file.split('/')[-1], args.config)
        phot_args.files = [(args.datadir[0] + field_run + '/cals_'
                            + ccd 
                            + 'temp/'
                            + file.split('/')[-1]).replace('.fits','.cat')]
        
        try:
            photom_correct(phot_args)

            if os.path.exists((args.datadir[0] + field_run + '/cals_' + ccd
                           + 'temp/'
                           + file.split('/')[-1]).replace(
                               '.fits','_corr.csv')) == False:
                cals.append(np.nan)
                cal_std.append(np.nan)
                continue
        except:
            print('Photometry correct did no work, moving on ...')

            cals.append(np.nan)
            cal_std.append(np.nan)
            continue
        df = pd.read_csv((args.datadir[0] + field_run + '/cals_' + ccd 
                          + 'temp/'
                          + file.split('/')[-1]).replace('.fits','_corr.csv'))
        
        mags = []
        for X,Y in zip(Xs,Ys):
            distances = [np.sqrt((df_X-X)**2 + (df_Y-Y)**2)
                         for df_X,df_Y in 
                         zip(np.array(df['X_IMAGE']),np.array(df['Y_IMAGE']))]
            mindex = np.argmin(distances)
            mags.append(df.iloc[mindex]['MAG'])
            
        mags = np.sort(mags)[::-1]
        calibration = np.median(-2.5*np.log10(fluxes) - mags)

        cals.append(calibration)
        stdev = np.std(-2.5*np.log10(fluxes) - mags)
        cal_std.append(stdev)
    cal_df = pd.DataFrame(np.transpose(np.array([files,cals,cal_std])),
                          columns=['file','zeropoint','zeropoint_err'])
    if os.path.exists(args.datadir[0] + field_run + '/cals_' + ccd 
                      + 'temp/')==False:
            os.makedirs(args.datadir[0] + field_run + '/cals_' + ccd 
                        + 'temp/')
    cal_df.to_csv(args.datadir[0] + field_run + '/cals/cals_' + ccd + 
                  '.csv',index=False)
    os.system('rm -r ' + args.datadir[0] + field_run + '/cals_' + ccd 
              + 'temp/')
