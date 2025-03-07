import pandas as pd
import numpy as np
import argparse
import os
from datastats.run_sourceextractor import run_sourceExtractor

def read_cat(file):
    with open(file,'r') as f:
        string = f.read()
        sexcolumns = [header.split()[1] for header in np.append(string.split('#')[1:-1],string.split('#')[-1].split('\n')[0])]
        f.close()
                                     
    cat = np.genfromtxt(file, unpack=True, filling_values=np.nan,
                        invalid_raise=False)
    cat = pd.DataFrame(data=np.transpose(cat), columns=sexcolumns)
    return cat

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
                        help='CCD number of the image.')
    args = parser.parse_args()
    
    print('Running SExtractor ... ')
    run_sourceExtractor(args.file,
                        sextractorloc='sex',
                        psfexloc='psfex',
                        verbose=False,quietmode=False,
                        debugmode=False,
                        spreadmodel=True,
                        detect_thresh=1.1)
    print('Done! Finding galaxies ... ')
    
    catname = args.file[0].split('/')[-1].replace('.fits','.cat')
    cat = read_cat(catname)
    
    xmax = np.max(cat.X_IMAGE)
    xmin = np.min(cat.X_IMAGE)
    ymax = np.max(cat.Y_IMAGE)
    ymin = np.min(cat.Y_IMAGE)
    
    cat = cat[(cat.X_IMAGE < xmax - 100) & (cat.X_IMAGE > xmin + 100) &
              (cat.Y_IMAGE < ymax - 100) & (cat.Y_IMAGE > ymin + 100)]
    
    galaxies = cat[(cat.SPREAD_MODEL > 0.01) & (cat.FLUX_MODEL > 500.)]
    galaxies = galaxies.sample(frac=1)
    
    nums_gone = None
    for num,ra,dec in zip(galaxies.NUMBER,galaxies.X_WORLD,galaxies.Y_WORLD):
        if np.isin(num,nums_gone):
            continue
        dist = np.sqrt((ra - galaxies.X_WORLD)**2 + (dec - galaxies.Y_WORLD)**2)
        mask = (galaxies.NUMBER != num) & (dist*3600. < 20.)
      
        if len(galaxies[mask]) == 0:
            continue
     
        nums_gone = pd.concat([nums_gone,galaxies[mask].NUMBER])
        galaxies = galaxies[mask == False]
    print(galaxies)
    print(nums_gone) 
    galaxies = galaxies.reset_index(drop=True)
    galaxies = galaxies.rename(columns={'X_WORLD':'ra',
                                        'Y_WORLD':'dec'})
    galaxies[['ra','dec']].to_csv('gals_' + args.ccd + '.csv',
                                                   index=False)
    galaxies[['ra','dec']].to_csv('gals_' + args.ccd + 
                                                   '_ds9check.txt',
                                  index=False,sep=' ',header=False)
    
    os.system('rm ' + catname)
    
    

