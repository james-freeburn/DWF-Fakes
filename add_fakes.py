import pandas as pd
import numpy as np
import astropy.units as u
import glob
import os
from astropy.wcs import WCS
from scipy.special import erf
from astropy.io import fits
import afterglowpy as grb
from astropy.cosmology import Planck18 as cosmo
import dask.dataframe as dd
from datastats.calculate_FWHM import calculate_FWHM
from astropy.modeling.functional_models import Moffat2D

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

def HandB_SFR(redshifts):
    a,b,c,d = 0.0170, 0.13, 3.3, 5.3
    return (a+b*redshifts)/(1 + (redshifts/c)**d)

def GRB_parameters(rng,nevents,t,shorts = 0):
    # Generating GRB times and light curve coverage.
    if shorts == 0:
        tGRB = rng.integers(-np.max(t) + 3, 1440, nevents)
    else:
        tGRB = np.concatenate([rng.uniform(-np.max(t) + 3, 1440, nevents - shorts),
                              (t[rng.integers(0, len(t)-1, shorts)] - np.max(t))/60.])
    short = np.concatenate([[False]*(nevents-shorts),[True]*shorts])

    theta_wings = rng.uniform(0.1,np.pi/2,nevents)
    n0s = 10**rng.normal(1.0, 1.0,nevents)
    b_vals = rng.uniform(0.1,6.0,nevents)
    ps = []
    ep_Bs = []
    ep_es = []
    for i in range(nevents):
        val_e = 1.0
        val_B = 1.0
        val_p = 1.0
        while val_e >= 1.0:
            val_e = 10**rng.uniform(np.log10(0.15), 0.0)
        ep_es.append(val_e)
        while val_B >= 1.0:
            val_B = 10**rng.normal(-4.0, 1.0)
        ep_Bs.append(val_B)
        while val_p < 2.0:
            val_p = rng.normal(2.21, 0.36)
        ps.append(val_p)
    ep_Bs = np.array(ep_Bs)
    ep_es = np.array(ep_es)
    ps = np.array(ps)
    mu, sigma = 52.5, 1.05
    Eiso_vals = 10**rng.normal(mu, sigma, nevents)
    # Generating viewing angles based on solid angle distributions.
    theta = np.linspace(0.0, np.pi/2, 100000)
    omega = 4*np.pi*np.sin(0.5*theta)**2
    probability_density = np.array([((omega[i] - omega[i-1])/
                                     (theta[i] - theta[i-1]))
                                    for i in range(1,len(theta))])/omega[-1]
    thetaObs_vals = []
    for i in range(nevents):
        roll = 1.0
        likelihood = probability_density[0]
        while (roll > likelihood):
            random_index = rng.integers(0,len(theta[theta < theta_wings[i]])-2)
            val = theta[random_index]
            likelihood = probability_density[random_index]
            roll = rng.uniform(0.0,1.0)
        thetaObs_vals.append(val)

    # Placing GRBs at redshifts based on comoving volumes and SFH.
    zs,d_Ls = GRB_redshifts(rng, nevents)

    return pd.DataFrame(np.transpose([thetaObs_vals,
                                      theta_wings,
                                      b_vals,
                                      zs,
                                      d_Ls,
                                      n0s,
                                      ps,
                                      ep_Bs,
                                      ep_es,
                                      Eiso_vals,
                                      tGRB,
                                      short]),
                        columns=['theta_v',
                                 'theta_w',
                                 'b',
                                 'z',
                                 'd_L',
                                 'n0',
                                 'p',
                                 'epsilon_B',
                                 'epsilon_e',
                                 'Eiso',
                                 'tGRB',
                                 'short'])

def GRB_redshifts(rng,nevents):
    redshifts = np.linspace(0.01,10.0,100000)
    SFH = HandB_SFR(redshifts)
    V = cosmo.comoving_volume(redshifts)
    volume_element = np.array([((V[i] - V[i-1])/
                                (redshifts[i]-redshifts[i-1])).value
                               for i in range(1,len(redshifts))])/V[-1].value
    probability = SFH[1:]*volume_element
    probability_density = probability/np.trapz(probability,redshifts[1:])
    zs = []
    for i in range(nevents):
        roll = 1.0
        likelihood = probability_density[0]
        while (roll > likelihood):
            random_index = rng.integers(0,len(redshifts)-2)
            z = redshifts[random_index]
            likelihood = probability_density[random_index]
            roll = rng.uniform(0.0,1.0)
        zs.append(z)
    d_Ls = cosmo.luminosity_distance(np.array(zs)).to(u.cm).value
    return zs,d_Ls

def check_detectability(row,field_run,names,t_mins):
    d = {}
    
    # For convenience, place arguments into a dict.
    Z = {'jetType':     grb.jet.PowerLaw,     # Power Law with Core
         'specType':    0,                  # Basic Synchrotron Spectrum
         'thetaObs':    row.theta_v,   # Viewing angle in radians
         'E0':          row.Eiso, # Isotropic-equivalent energy in erg
         'thetaCore':   0.1,    # Half-opening angle in radians
         'thetaWing':   row.theta_w,    # Setting thetaW
         'b':           row.b,
         'n0':          row.n0,    # circumburst density in cm^{-3}
         'p':           row.p,    # electron energy distribution index
         'epsilon_e':   row.epsilon_e,    # epsilon_e
         'epsilon_B':   row.epsilon_B,   # epsilon_B
         'xi_N':        1.0,    # Fraction of electrons accelerated
         'd_L':         row.d_L, # Luminosity distance in cm
         'z':           row.z}   # redshift
    
    wavel = 473*10**(-9)
    c = 299792458
    #g-band central frequency
    nu =  c/wavel
    
    t = (t_mins + row.tGRB)*60

    if -2.5*np.log10(np.max(
            grb.fluxDensity(
                np.array(range(1,10))*60., nu, **Z))*10**(-3)) + 8.9 > 22.25:
        d['detectable'] = False
    else:
        Fnu_GRB = grb.fluxDensity(t[t>0], nu, **Z)
        Fnu_before = np.array([np.nan for time in t[t<=0]])
        Fnu = np.append(Fnu_before,Fnu_GRB)
        gmag = -2.5*np.log10(np.array(Fnu)*10**(-3)) + 8.9

        detectability = np.min(gmag[np.isnan(gmag) == False]) < 22.25
        d['detectable'] = detectability
        if detectability:
            lc = pd.DataFrame(np.transpose([names,t,gmag]),
                              columns=['names','t','gmag'])
            lc.to_csv('light_curves/' + field_run + '/' + str(int(row.name)) + '.csv',index=False)
        
    return pd.Series(d, dtype=object)

def generate_afterglows(rng,nevents,t,shorts=0):
    # Generating GRB parameters.
    meta_data = GRB_parameters(rng,nevents,t,shorts=shorts)
    meta_data_dd = dd.from_pandas(meta_data,npartitions=1)
    detectable_arr = meta_data_dd.apply(check_detectability,axis=1,
                                        field_run=field_run,t_mins=t,
                                        names=fitsnames,
                                     meta={'detectable':bool}).compute()
    meta_data = meta_data.assign(detectable = detectable_arr)
    
    return meta_data[meta_data['detectable']]

def distribute_events(rng,afterglows,gal_coords,fitsfile):
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
    
    coords = np.array([rng.uniform(ra_min,ra_max,len(afterglows)),
                       rng.uniform(dec_min,dec_max,len(afterglows))])
    afterglows = afterglows.assign(ra = coords[0],dec = coords[1])
    
    n_gal_events = int(np.floor(len(afterglows)/2))
    if n_gal_events > len(gal_coords):
        n_gal_events = len(gal_coords)

    indices = rng.choice(np.array(gal_coords.index),n_gal_events,replace=False)
    for i in range(n_gal_events):
        afterglows.iloc[i,-2] = gal_coords['ra'][indices[i]]
        afterglows.iloc[i,-1] = gal_coords['dec'][indices[i]]
        
    return afterglows

def add_fakes(afterglows,light_curves,names,fitsfiles,cals):
    for i in range(len(fitsfiles)):
        if os.path.exists(names[i].replace('.fits','_fakes.fits')):
            continue
  
        wcs = WCS(fitsfiles[i][0].header)
        data = fitsfiles[i][0].data

        fwhm=np.abs(calculate_FWHM(
                     [names[i]],
                     sextractorloc='sex',
                     verbose=False,
                     quietmode=True)[0])
        
        Xs,Ys = wcs.wcs_world2pix(afterglows['ra'],afterglows['dec'],1)
        print(names[i],cals['zeropoint'][cals['file'] == names[i]])
        cal = float(cals['zeropoint'][cals['file'] == names[i]])
        mags = [float(light_curves[j]['gmag'][
            light_curves[j]['names'] == names[i]]) 
            for j in range(len(afterglows))]
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
    rng = np.random.default_rng(seed=12345)
    nevents = 1100
    shorts = 100
    field_run = 'FRB190711_09_2022'
    ccd = 22

    fakes = pd.read_csv('fakes.csv')
    cals = pd.read_csv('data/' + field_run + '_' + str(22) + '/cals.csv')
    
    if os.path.exists('light_curves/') == False:
        os.makedirs('light_curves/')
    added_df = None
    if os.path.exists('light_curves/' + field_run + '/') == False:
        os.makedirs('light_curves/' + field_run + '/')
        if len(glob.glob('light_curves/' + field_run + '/')) == 0:
            added_lcs = glob.glob('light_curves/' + field_run + '/')
            print('Found ' + str(len(added_lcs)) + ' added light curves!')
            added_df = pd.DataFrame({'theta_v':[np.nan]*len(added_lcs),
                                     'theta_w':[np.nan]*len(added_lcs),
                                     'b':[np.nan]*len(added_lcs),
                                     'z':[np.nan]*len(added_lcs),
                                     'd_L':[np.nan]*len(added_lcs),
                                     'n0':[np.nan]*len(added_lcs),
                                     'p':[np.nan]*len(added_lcs),
                                     'epsilon_B':[np.nan]*len(added_lcs),
                                     'epsilon_e':[np.nan]*len(added_lcs),
                                     'Eiso':[np.nan]*len(added_lcs),
                                     'tGRB':[np.nan]*len(added_lcs),
                                     'detectable':[True]*len(added_lcs),
                                     'short':[False]*len(added_lcs),
                                     'lc_name':added_lcs})
    print("Reading in fits files ...")
    fitsnames = glob.glob('data/' + field_run + '_' + str(ccd) +
                      '/*/c4d_*_ooi_g_v1/c4d_*_ooi_g_v1_ext' 
                      + str(ccd) + '.fits')
    fitsnames = np.sort(np.array(fitsnames))
    fitsfiles = [fits.open(file) for file in fitsnames]
    print("Done!")
    MJDs = np.array([file[0].header['MJD-OBS'] for file in fitsfiles])
    t = (MJDs - MJDs[0])*24*60
    
    gal_coords = pd.read_csv('data/'+ field_run + '_' + str(ccd) + '/gals.csv')
    
    if os.path.exists('data/'+ field_run + '_' + str(ccd) + '/afterglows.csv'):
        afterglows = pd.read_csv('data/'+ field_run + '_' + str(ccd) + 
                                 '/afterglows.csv')
    else:
        print("Generating afterglows ...")
        afterglows = generate_afterglows(rng,nevents,t,shorts=shorts)
        afterglows = afterglows.assign(lc_name=afterglows.index.astype(str) + '.csv')
        afterglows = pd.concat([afterglows,added_df])
        afterglows = distribute_events(rng,afterglows,gal_coords,fitsfiles[0])
        afterglows.to_csv('data/'+ field_run + '_' + str(ccd) + 
                          '/afterglows.csv',index=False)
        print("Done!")
    
    light_curves = [pd.read_csv('light_curves/' + field_run + '/' + lc_name) for lc_name in afterglows.lc_name]
    
    print("Adding afterglows to images ...")
    add_fakes(afterglows,light_curves,fitsnames,fitsfiles,cals)
    print("All Done!")
