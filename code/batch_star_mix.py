#!/usr/bin/env python
import os

ms = list(range(104, 124))

os.chdir(os.path.expandvars('$xveganx/sf/V827Tau/'))

for m in ms:
    print(m)
    os.chdir("m{:03d}/output/mix_emcee/run01/".format(m))
    os.system('time $xveganx/code/star_mix_beta.py --samples=5000 --incremental_save=100 > run.out')
    os.chdir(os.path.expandvars('$xveganx/sf/V827Tau/'))
