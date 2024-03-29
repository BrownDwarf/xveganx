#!/usr/bin/env python
import os
import yaml
import numpy as np
import h5py

ms = list(range(105, 125))
#ms = list(range(105, 106))

os.chdir(os.path.expandvars('$xveganx/sf/V827Tau/'))

for m in ms:

    dat_name = os.path.expandvars('$xveganx/data/IGRINS/reduced/V827_IGRINS_20141121_m{:03d}.hdf5'.format(m))
    path_out = 'm{:03d}/'.format(m)
    sf_out = 'm{:03d}/config.yaml'.format(m)

    f = open("m104/output/mix_emcee/run01/config.yaml")
    config = yaml.load(f)
    f.close()

    f=h5py.File(dat_name, "r")
    wls = f['wls'][:]
    f.close()

    config['data']['files'] = ['$xveganx/data/IGRINS/reduced/V827_IGRINS_20141121_m{:03d}.hdf5'.format(m)]
    config['grid']['hdf5_path'] = '$xveganx/sf/V827Tau/m{:03d}/libraries/PHOENIX_IGRINS_m{:03d}_Teff2700-4500.hdf5'.format(m,m)
    lb, ub = int(np.floor(wls[0])), int(np.ceil(wls[-1]))

    config['grid']['wl_range'] = [lb, ub]
    config['PCA']['path'] = '$xveganx/sf/V827Tau/m{:03d}/PHOENIX_IGRINS_H_PCA_Teff2700-4500.hdf5'.format(m)
    config['data']['instruments'] =['IGRINS_H']
    config['Theta_priors']= '$xveganx/sf/V827Tau/m{:03d}/user_prior.py'.format(m)

    os.makedirs(path_out, exist_ok=True)
    with open(sf_out, mode='w') as outfile:
        outfile.write(yaml.dump(config))
        print('wrote to {}'.format(path_out))

#for m in ms:
#    os.chdir("m{:03d}".format(m))
#    os.system('mkdir libraries &')
#    os.system('$Starfish/scripts/grid.py --create > grid.out &')
#    os.system('cp $xveganx/sf/V827Tau/m104/user_prior.py .')
#    os.chdir("..")