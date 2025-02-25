import numpy as np
import afterglowpy as grb
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import dask.dataframe as dd

def generate_GRBs(rng,nevents):
    mu_gamma = 1.95
    sigma_gamma = 0.65
    sigma_theta_j = 0.3
    
    m = 2.5
    q = 1.45
    
    log_Gamma_0s = rng.normal(mu_gamma,sigma_gamma,nevents)
    log_theta_js = np.array([rng.normal((-1./m)*log_Gamma_0 + q,sigma_theta_j) 
                             for log_Gamma_0 in log_Gamma_0s])
    Gamma_0s = 10**log_Gamma_0s
    theta_js = (np.pi/180.)*10**log_theta_js
    mask = (Gamma_0s > 1.) & (Gamma_0s < 8e3) & (theta_js < np.pi/2.) 
    
    return Gamma_0s[mask],theta_js[mask]

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def GRBFR(z):
    R = (0.00157 + 0.118*z)/(1+(z/3.23)**4.66)
    e = (1+z)**1.7
    return R*e

def population_synthesis(rng,nevents,cosmo,
                         E_gamma_dash = 1.5e48,E_p_dash = 1.5):    

    print('\tGenerating Data ... ')
    Gamma_0s,theta_js = generate_GRBs(rng,nevents)
    nevents = len(Gamma_0s)
    
    print('\tGenerating theta_v values ... ')
    # Generating probability density function for viewing angle.
    theta = np.linspace(0.0, np.pi/2, 1000)
    probability_density = np.sin(theta)
    probability_density = probability_density/np.sum(probability_density)
    thetaObs_vals = rng.choice(theta, nevents, p=probability_density)
    
    print('\tCalculating params ... ')

    jets = [1,4]
    df = pd.DataFrame([])
    df = df.assign(Gamma_0=Gamma_0s,
                   theta_j=theta_js,
                   theta_v=thetaObs_vals,
                   jet=rng.choice(jets,nevents),
                   Beta_0 = (1.-(1./Gamma_0s**2.))**0.5,
                   E_gamma = E_gamma_dash*Gamma_0s)
    
    df = df.assign(GRB_id = df.index,
                   E_p = E_p_dash*5.*df.Gamma_0/(5.-2.*df.Beta_0),
                   Eiso = [E_gamma/(1-np.cos(theta_j)) 
                           if 1./Gamma_0 < np.sin(theta_j)
                           else E_gamma*(1+Beta_0)*Gamma_0**2
                           for E_gamma,Gamma_0,Beta_0,theta_j 
                           in zip(df.E_gamma,df.Gamma_0,df.Beta_0,df.theta_j)
                           ])
    
    print('\tAssigning redshifts ...')
    # Placing GRBs at redshifts based on comoving volumes and SFH.
    zs,d_Ls = GRB_redshifts(rng, len(df), cosmo)
    df = df.assign(z=zs,d_L=d_Ls)

    # Generating GRB times and light curve coverage.
    df = df.assign(n0 = rng.uniform(0.1,30.,len(df)),
                   b = rng.uniform(0.0,3.0,len(df)),
                   epsilon_B = [0.008]*len(df),
                   epsilon_e = [0.02]*len(df),
                   p = [2.3]*len(df),
                   theta_w = [rng.uniform(theta_j,np.pi/2) 
                              for theta_j in df.theta_j]
                   ).reset_index(drop=True)
    # Uniform distribution of theta_wing values
    
    return df[['theta_v',
              'jet',
              'theta_w',
              'theta_j',
              'b',
              'z',
              'd_L',
              'n0',
              'p',
              'epsilon_B',
              'epsilon_e',
              'Eiso',
              'Gamma_0',
              'E_p']]

def GRB_redshifts(rng,nevents,cosmo):
    redshifts = np.linspace(0.01,10.0,1000)
    V = cosmo.comoving_volume(redshifts)
    volume_element = np.gradient(V,redshifts)
    
    probability = GRBFR(redshifts)*volume_element.value/(1+redshifts)
    probability_density = probability/np.sum(probability)
    
    zs = rng.choice(redshifts, nevents, p=probability_density)
    d_Ls = cosmo.luminosity_distance(np.array(zs)).to(u.cm).value
    return zs,d_Ls

def check_detectability(row,rng,names,MJDs,field_run):
    d = {}
    
    # For convenience, place arguments into a dict.
    Z = {'jetType':     row.jet,
         'specType':    0,                  # Basic Synchrotron Spectrum
         'thetaObs':    row.theta_v,   # Viewing angle in radians
         'E0':          row.Eiso, # Isotropic-equivalent energy in erg
         'thetaCore':   row.theta_j,    # Half-opening angle in radians
         'thetaWing':   row.theta_w,    # Setting thetaW
         'b':           row.b,
         'n0':          row.n0,    # circumburst density in cm^{-3}
         'p':           row.p,    # electron energy distribution index
         'epsilon_e':   row.epsilon_e,    # epsilon_e
         'epsilon_B':   row.epsilon_B,   # epsilon_B
         'xi_N':        1.0,    # Fraction of electrons accelerated
         'd_L':         row.d_L, # Luminosity distance in cm
         'z':           row.z}   # redshift

    t = (MJDs - MJDs[0])*24.*60.*60.
    if row.short:
        d['tGRB'] = t[rng.integers(0, len(t)-1)] - np.max(t)
    else:
        d['tGRB'] = rng.uniform(-24.*60.*60., np.max(t))
    t = t - d['tGRB']
    wavel = 473*10**(-9)
    c = 299792458
    # g-band central frequency
    nu =  c/wavel
    
    if -2.5*np.log10(np.max(grb.fluxDensity(
            np.geomspace(1,500,10)*60.,nu, **Z))*10**(-3)) + 8.9 > 23.:
        d['detectable'] = False
        d['peak_mag'] = np.nan
    else:
        Fnu_GRB = grb.fluxDensity(t[t>0], nu, **Z)
        Fnu_before = np.array([np.nan for time in t[t<=0]])
        Fnu = np.append(Fnu_before,Fnu_GRB)

        gmag = -2.5*np.log10(np.array(Fnu)*10**(-3)) + 8.9
        d['detectable'] = np.min(gmag[np.isnan(gmag) == False]) < 23.
        d['peak_mag'] = np.min(gmag[np.isnan(gmag) == False])

        if d['detectable']:
            lc = pd.DataFrame(np.transpose([names,MJDs,t,gmag]),
                              columns=['names','MJD','t','gmag'])
            lc.to_csv(field_run + str(int(row.name)) + '.csv',index=False)

    return pd.Series(d, dtype=object)

def generate_afterglows(rng,nevents,fitsnames,MJDs,directory,shorts=0):
    # Generating GRB parameters.
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    GRB_population = population_synthesis(rng,nevents,cosmo)
    GRB_population = GRB_population.assign(
        short = np.concatenate([[False]*(len(GRB_population)-shorts),
                                [True]*shorts]))
    GRB_population_dd = dd.from_pandas(GRB_population,npartitions=1)
    detectable_arr = GRB_population_dd.apply(check_detectability,axis=1,
                                             rng=rng,
                                             names=fitsnames,
                                             MJDs=MJDs,
                                             field_run=directory,
                                     meta={'tGRB':float,
                                           'detectable':bool,
                                           'peak_mag':float}).compute()
    
    GRB_population = GRB_population.assign(
        tGRB = detectable_arr.tGRB,
        DWF_afterglow = detectable_arr.detectable,
        afterglow_peakmag = detectable_arr.peak_mag)
    
    return GRB_population[GRB_population['DWF_afterglow']]
