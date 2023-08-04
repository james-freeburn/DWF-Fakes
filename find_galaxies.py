import pandas as pd
import numpy as np
import argparse
from datastats.run_sourceextractor import run_sourceextractor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Find galaxies in a fits \
                                     image and output a csv with their \
                                     locations.')
    parser.add_argument('-f', '--file',
                        type=str,
                        nargs=1,
                        help='File to run on.')
    parser.add_argument('-c', '--ccd',
                        type=str,
                        nargs=1,
                        help='CCD number of the image.')
    args = parser.parse_args()

    run_sourceextractor(args.file,
                        sextractorloc='sex',
                        psfexloc='psfex',
                        verbose=False,quietmode=False,
                        debugmode=False,
                        spreadmodel=True,
                        savecats='./',
                        detect_thresh=1.1)
    
    catname = args.file[0].replace('.fits','.cat')
    columns=['NUMBER','X_IMAGE','Y_IMAGE','X_WORLD','Y_WORLD','FLUX_MODEL',
             'FLUXERR_MODEL','MAG_MODEL','MAGERR_MODEL','FWHM_IMAGE',
             'FWHM_WORLD','ELLIPTICITY','SPREAD_MODEL']
    cat = pd.read_csv(catname,delim_whitespace=True,names=columns,header=None)
    
    galaxies = cat[(cat.SPREAD_MODEL > 0.01) & (cat.FLUX_MODEL > 500.)]
    galaxies = galaxies.sample(frac=1)
    
    nums_gone = None
    for num,ra,dec in zip(galaxies.NUMBER,galaxies.X_WORLD,galaxies.Y_WORLD):
        if np.isin(num,nums_gone):
            continue
        dist = np.sqrt((ra - galaxies.X_WORLD)**2 + (dec - galaxies.Y_WORLD)**2)
        mask = (galaxies.NUMBER != num) & (dist*3600. < 10.)
      
        if len(galaxies[mask]) == 0:
            continue
     
        nums_gone = pd.concat([nums_gone,galaxies[mask].NUMBER])
        galaxies = galaxies[mask == False]
        
    galaxies = galaxies.reset_index(drop=True)
    galaxies = galaxies.rename(columns={'X_WORLD':'ra',
                                        'Y_WORLD':'dec'})
    galaxies[['ra','dec']].to_csv('gals_ext27.csv',index=False)
    galaxies[['ra','dec']].to_csv('gals_ext27_ds9check.txt',index=False,
                                  sep=' ',header=False)
    
    
    

