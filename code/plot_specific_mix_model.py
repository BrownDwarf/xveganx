#!/usr/bin/env python

# All of the argument parsing is done in the `parallel.py` module.

import multiprocessing
import time
import numpy as np
import Starfish
from Starfish.model import ThetaParam, PhiParam

import argparse
parser = argparse.ArgumentParser(prog="plot_many_mix_models.py", description="Plot many mixture models.")
parser.add_argument("--ff", type=int, default=3, help="Number of fill factor models to assume")
parser.add_argument("--config", action='store_true', help="Use config file instead of emcee.")
parser.add_argument("--static", action="store_true", help="Make a static figure of one draw")
parser.add_argument("--animate", action="store_true", help="Make an animation of many draws from the two components.")
parser.add_argument("--OG", action="store_true", help="The Original Gangster version, clunky and all.")
args = parser.parse_args()

import os

import matplotlib.pyplot as plt


import os

import Starfish.grid_tools
from Starfish.spectrum import DataSpectrum, Mask, ChebyshevSpectrum
from Starfish.emulator import Emulator
import Starfish.constants as C
from Starfish.covariance import get_dense_C, make_k_func, make_k_func_region

from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet
from astropy.stats import sigma_clip

import gc
import logging

from itertools import chain
#from collections import deque
from operator import itemgetter
import yaml
import shutil
import json

from star_base import Order as OrderBase
from star_base import SampleThetaPhi as SampleThetaPhiBase

Starfish.routdir = ""

# list of keys from 0 to (norders - 1)
order_keys = np.arange(1)
DataSpectra = [DataSpectrum.open(os.path.expandvars(file), orders=Starfish.data["orders"]) for file in Starfish.data["files"]]
# list of keys from 0 to (nspectra - 1) Used for indexing purposes.
spectra_keys = np.arange(len(DataSpectra))

#Instruments are provided as one per dataset
Instruments = [eval("Starfish.grid_tools." + inst)() for inst in Starfish.data["instruments"]]


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    Starfish.routdir), level=logging.DEBUG, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

class Order(OrderBase):

    def initialize(self, key):
        OrderBase.initialize(self, key)
        self.flux_scalar2 = None
        self.mus2, self.C_GP2 = None, None
        self.Omega2 = None

    def evaluate(self):
        '''
        Return the lnprob using the current version of the C_GP matrix, data matrix,
        and other intermediate products.
        '''

        self.lnprob_last = self.lnprob

        X = (self.chebyshevSpectrum.k * self.flux_std * np.eye(self.ndata)).dot(self.eigenspectra.T)

        part1 = self.Omega**2 * self.flux_scalar**2 * X.dot(self.C_GP.dot(X.T))
        part2 = self.Omega2**2 * self.flux_scalar2**2 * X.dot(self.C_GP2.dot(X.T))
        part3 = self.data_mat

        #CC = X.dot(self.C_GP.dot(X.T)) + self.data_mat
        CC = part1 + part2 + part3
        np.save('CC.npy', CC)
        try:
            factor, flag = cho_factor(CC)
        except np.linalg.linalg.LinAlgError:
            print("Spectrum:", self.spectrum_id, "Order:", self.order)
            self.CC_debugger(CC)
            raise

        try:
            model1 = self.Omega * self.flux_scalar *(self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus))
            model2 = self.Omega2 * self.flux_scalar2 * (self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus2))
            net_model = model1 + model2
            R = self.fl - net_model

            logdet = np.sum(2 * np.log((np.diag(factor))))
            self.lnprob = -0.5 * (np.dot(R, cho_solve((factor, flag), R)) + logdet)

            self.logger.debug("Evaluating lnprob={}".format(self.lnprob))
            return self.lnprob

        # To give us some debugging information about what went wrong.
        except np.linalg.linalg.LinAlgError:
            print("Spectrum:", self.spectrum_id, "Order:", self.order)
            raise


    def update_Theta(self, p):
        OrderBase.update_Theta(self, p)
        self.emulator.params = np.append(p.teff2, p.grid[1:])
        self.mus2, self.C_GP2 = self.emulator.matrix
        self.flux_scalar2 = self.emulator.absolute_flux
        self.Omega2 = 10**p.logOmega2

    def draw_save(self):
        '''
        Return the lnprob using the current version of the C_GP matrix, data matrix,
        and other intermediate products.
        '''

        self.lnprob_last = self.lnprob

        X = (self.chebyshevSpectrum.k * self.flux_std * np.eye(self.ndata)).dot(self.eigenspectra.T)

        model1 = self.Omega * self.flux_scalar *(self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus))
        model2 = self.Omega2 * self.flux_scalar2 * (self.chebyshevSpectrum.k * self.flux_mean + X.dot(self.mus2))
        net_model = model1 + model2
        model_out = net_model

        return model_out

class SampleThetaPhi(Order, SampleThetaPhiBase):
    pass


# Run the program.

model = SampleThetaPhi(debug=True)

model.initialize((0,0))

def lnlike(p):
    # Now we can proceed with the model
    try:
        #pars1 = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5])
        pars1 = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5], teff2=p[6], logOmega2=p[7])
        model.update_Theta(pars1)
        # hard code npoly=3 (for fixc0 = True with npoly=4)
        #pars2 = PhiParam(0, 0, True, p[6:9], p[9], p[10], p[11])
        pars2 = PhiParam(0, 0, True, p[8:11], p[11], p[12], p[13])
        model.update_Phi(pars2)
        lnp = model.evaluate()
        return lnp
    except C.ModelError:
        model.logger.debug("ModelError in stellar parameters, sending back -np.inf {}".format(p))
        return -np.inf


# Must load a user-defined prior
try:
    sourcepath_env = Starfish.config['Theta_priors']
    sourcepath = os.path.expandvars(sourcepath_env)
    with open(sourcepath, 'r') as f:
        sourcecode = f.read()
    code = compile(sourcecode, sourcepath, 'exec')
    exec(code)
    lnprior = user_defined_lnprior
    print("Using the user defined prior in {}".format(sourcepath_env))
except:
    print("Don't you want to use a user defined prior??")
    raise

# Insert the prior here
def lnprob(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p)


# Run the program.

model = SampleThetaPhi(debug=True)

model.initialize((0,0))

def lnprob_all(p):
    pars1 = ThetaParam(grid=p[0:3], vz=p[3], vsini=p[4], logOmega=p[5], teff2=p[6], logOmega2=p[7])
    model.update_Theta(pars1)
    # hard code npoly=3 (for fixc0 = True with npoly=4)
    #pars2 = PhiParam(0, 0, True, p[6:9], p[9], p[10], p[11])
    pars2 = PhiParam(0, 0, True, p[8:11], p[11], p[12], p[13])
    model.update_Phi(pars2)
    junk = model.evaluate()
    draw = model.draw_save()
    return draw

draws = []

try:
    ws = np.load("emcee_chain.npy")
    burned = ws[:, -200:,:]
except:
    ws = np.load("temp_emcee_chain.npy")
    max_save = ws.any(axis=(0,2)).sum()
    burned = ws[:, max_save-200:max_save,:]

xs, ys, zs = burned.shape
fc = burned.reshape(xs*ys, zs)

nx, ny = fc.shape


#Colorbrewer bands
s3 = '#fee6ce'
s2 = '#fdae6b'
s1 = '#e6550d'

wl = model.wl
data = model.fl

import pandas as pd
import json

if args.OG:

    median_vz_shift = np.median(fc[:, 3])
    dlam = median_vz_shift/299792.0*np.median(wl)

    # Get the line list of strong lines in Arcturus

    all_ll = pd.read_csv('/Users/obsidian/GitHub/ApJdataFrames/data/Rayner2009/tbl7_clean.csv')
    all_ll['wl_A'] = all_ll.wl*10000.0

    ll = all_ll[ (all_ll.wl_A > np.min(wl)) & (all_ll.wl_A < np.max(wl)) ]
    ll = ll.reset_index()


    # Sort the flatchain by fill factor:
    ff = 10**fc[:, 7]/(10**fc[:, 7]+10**fc[:, 5])
    inds_sorted = np.argsort(ff)
    fc_sorted = fc[inds_sorted]
    # If we use 8000 samples, the 5th and 95th percentile samples are at:
    ind_lo = 400 #0.05*8000
    ind_med = 4000 #0.50*8000
    ind_hi = 7600 #0.95*8000

    df_out = pd.DataFrame({'wl':wl, 'data':data})

    # Low end:
    ps_lo = fc_sorted[ind_lo]
    print(ps_lo)
    df_out['model_comp05'] = lnprob_all(ps_lo)

    pset1 = ps_lo.copy()
    pset1[5] = -20
    df_out['model_cool05'] = lnprob_all(pset1)
    pset2 = ps_lo.copy()
    pset2[7] = -20
    df_out['model_hot05'] = lnprob_all(pset2)

    # Middle:
    ps_med = fc_sorted[ind_med]
    df_out['model_comp50'] = lnprob_all(ps_med)

    pset1 = ps_med.copy()
    pset1[5] = -20
    df_out['model_cool50'] = lnprob_all(pset1)
    pset2 = ps_med.copy()
    pset2[7] = -20
    df_out['model_hot50'] = lnprob_all(pset2)

    # Hi end:
    ps_hi = fc_sorted[ind_hi]
    df_out['model_comp95'] = lnprob_all(ps_hi)

    pset1 = ps_hi.copy()
    pset1[5] = -20
    df_out['model_cool95'] = lnprob_all(pset1)
    pset2 = ps_hi.copy()
    pset2[7] = -20
    df_out['model_hot95'] = lnprob_all(pset2)

    df_out.to_csv('models_ff-05_50_95.csv', index=False)


if args.config:
    df_out = pd.DataFrame({'wl':wl, 'data':data})

    with open('s0_o0phi.json') as f:
        s0phi = json.load(f)

    psl = (Starfish.config['Theta']['grid']+
      [Starfish.config['Theta'][key] for key in ['vz', 'vsini', 'logOmega', 'teff2', 'logOmega2']] +
      s0phi['cheb'] +
      [s0phi['sigAmp']] + [s0phi['logAmp']] + [s0phi['l']])

    ps = np.array(psl)
    df_out['model_composite'] = lnprob_all(ps)

    pset1 = ps.copy()
    pset1[5] = -20
    df_out['model_cool50'] = lnprob_all(pset1)
    pset2 = ps.copy()
    pset2[7] = -20
    df_out['model_hot50'] = lnprob_all(pset2)

    df_out.to_csv('spec_config.csv', index=False)

if args.static:

    draws = []

    ws = np.load("emcee_chain.npy")

    burned = ws[:, 4997:5000,:]
    xs, ys, zs = burned.shape
    fc = burned.reshape(xs*ys, zs)

    nx, ny = fc.shape

    median_vz_shift = np.median(fc[:, 3])
    dlam = median_vz_shift/299792.0*np.median(wl)


    # Sort the flatchain by fill factor:
    fc_sorted = fc
    ind_med = 60 #Random

    df_out = pd.DataFrame({'wl':wl, 'data':data})

    # Middle:
    ps_med = fc_sorted[ind_med]
    df_out['model_comp50'] = lnprob_all(ps_med)

    df_out.to_csv('models_draw.csv', index=False)

if args.animate:
    from matplotlib import animation

    n_draws = 200
    rints = np.random.randint(0, nx, size=n_draws)
    ps_es = fc[rints]
    asi = ps_es[:, 4].argsort()
    ps_vals = ps_es[asi , :]

    draws = []

    for i in range(n_draws):
        ps = ps_vals[i]
        draw = lnprob_all(ps)
        draws.append(draw)

    """
    Matplotlib Animation Example

    author: Jake Vanderplas
    email: vanderplas@astro.washington.edu
    website: http://jakevdp.github.com
    license: BSD
    Please feel free to use and modify this, but keep the above information. Thanks!
    """

    import seaborn as sns
    sns.set_context('talk', font_scale=1.5)
    sns.set_style('ticks')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.step(wl, data, 'k', label='Data')
    ax.set_xlim(np.min(wl), np.max(wl))
    ax.set_xlabel(r"$\lambda (\AA)$")
    ax.set_ylim(0, 1.3*np.percentile(data, 95))
    #ax.set_yticks([])
    #ax.set_xticks([])

    # First set up the figure, the axis, and the plot element we want to animate
    line, = ax.plot([], [], color='#AA00AA', lw=2, label='Model')

    plt.legend(loc='upper right')

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return [line]

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(wl, draws[i])
        return [line]


    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('BD_IG_spec_anim.mp4', fps=10, dpi=300)
