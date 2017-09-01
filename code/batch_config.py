#!/usr/bin/env python
import os
import yaml
import numpy as np
import h5py

ms = ms = list(range(100, 112)) + list(range(113, 126))


os.chdir(os.path.expandvars('$xveganx/sf/Anon1/'))

os.getcwd()

for m in ms:

    dat_name = os.path.expandvars('$xveganx/data/IGRINS/reduced/Anon1_20141118_{:03d}.hdf5'.format(m))
    path_out = 'm{:03d}/'.format(m)
    sf_out = 'm{:03d}/config.yaml'.format(m)

    f = open("m112/config.yaml")
    config = yaml.load(f)
    f.close()

    f=h5py.File(dat_name, "r")
    wls = f['wls'][:]
    f.close()

    config['data']['files'] = ['$xveganx/data/IGRINS/reduced/Anon1_20141118_{:03d}.hdf5'.format(m)]
    config['grid']['hdf5_path'] = '$xveganx/sf/Anon1/m{:03d}/libraries/PHOENIX_IGRINS_m{:03d}_Teff2700-4500.hdf5'.format(m,m)
    lb, ub = int(np.floor(wls[0])), int(np.ceil(wls[-1]))

    config['grid']['wl_range'] = [lb, ub]
    config['PCA']['path'] = '$xveganx/sf/Anon1/m{:03d}/PHOENIX_IGRINS_H_PCA_Teff2700-4500.hdf5'.format(m)
    config['data']['instruments'] =['IGRINS_H']

    os.makedirs(path_out, exist_ok=True)
    with open(sf_out, mode='w') as outfile:
        outfile.write(yaml.dump(config))
        print('wrote to {}'.format(path_out))

for m in ms:
    os.chdir("m{:03d}".format(m))
    os.system('mkdir libraries &')
    os.system('$Starfish/scripts/grid.py --create > grid.out &')
    os.chdir("..")