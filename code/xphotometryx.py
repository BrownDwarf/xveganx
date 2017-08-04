import numpy as np
import pandas as pd
import math
import datetime as dt
from gatspy.periodic import LombScargle, LombScargleFast
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.time_series import multiterm_periodogram
from astroML.time_series import lomb_scargle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper', font_scale=1.4)
sns.set_style('ticks')

from scipy.signal import argrelmax

def flat_grankin08(grankin_fn):
    '''
    Returns a "flat" pandas DataFrame given a filename
    '''
    cols = ['HJD', 'Vmag', 'U_B', 'B_V', 'V_R']
    gr_data = pd.read_csv(grankin_fn, na_values=9.999,
                          names=cols, delim_whitespace=True)
    gr_data['Bmag']=gr_data['B_V']+gr_data['Vmag']
    gr_data['Rmag']=gr_data['Vmag']-gr_data['V_R']
    gr_data['Umag']=gr_data['U_B']+gr_data['Bmag']
    gr_data['Berr']=0.01
    gr_data['Verr']=0.01
    gr_data['Rerr']=0.01
    gr_data['Uerr']=0.05
    gr_data.drop(['U_B', 'B_V', 'V_R'], axis=1, inplace=True)
    gr_data['date_type'], gr_data['source'], gr_data['n_obs'] = 'HJD', 'Grankin et al. 2008', 1.0
    gr_data = gr_data.rename(columns={"HJD":"JD_like"})
    return gr_data

def clean_ASASSN(ASASSN_fn):
    '''
    Cleans an ASASSN file by grouping multiple visits, returns a DataFrame
    '''
    # Read in the data
    dat = pd.read_csv(ASASSN_fn, na_values=99.990)

    # Drop measurements with erroneous measurements
    bad_vals = dat.mag > 20.0
    dat.drop(dat.index[bad_vals], axis=0, inplace=True)

    # Find the difference in time between nearby measurements
    diff_df = pd.DataFrame({"a":dat.HJD[1:].values, "b":dat.HJD[0:-1].values, "c":dat.mag[1:].values})
    diff_df['d'] = diff_df.a - diff_df.b
    diff_df['d_min'] = diff_df.d * 24 * 60.0

    # Assign groups of measurements taken in 10 minute campaigns
    diff_df['big_jump'] = diff_df.d_min > 10
    diff_df['campaign'] = np.NAN
    diff_df.campaign[diff_df.big_jump] = np.arange(diff_df.big_jump.sum())
    diff_df.campaign.fillna(method='ffill', inplace=True)
    diff_df.dropna(inplace=True)
    grouped = diff_df.groupby('campaign')

    # Compute the mean and variance---with a floor---with each campaign
    means = grouped.aggregate({"a":np.mean, "c":np.mean})
    stddevs = grouped.aggregate({"c":np.std})
    n_obs = grouped.aggregate({"c":len})
    aggregated_data = pd.concat([means.rename(columns={'a':'HJD', 'c':'mean_mag'}),
                                 stddevs.rename(columns={'c':'stddev'}  ),
                                 n_obs.rename(columns={'c':'n_obs'})],  axis=1)

    aggregated_data.stddev[aggregated_data.stddev < 0.01] = 0.01
    aggregated_data.stddev[aggregated_data.stddev != aggregated_data.stddev ] = 0.01

    return aggregated_data.reset_index().drop(['campaign'], axis=1)


def flat_ASASSN(ASASSN_fn):
    '''
    Returns a "flat" pandas DataFrame given a cleaned, but un-flat DataFrame
    '''
    df = clean_ASASSN(ASASSN_fn)
    df['date_type'], df['source'] = 'HJD', 'ASASSN'
    df = df.rename(columns={"mean_mag":"Vmag", "stddev":"Verr", "HJD":"JD_like"})

    return df

@np.vectorize
def jd_to_date(jd):
    """
    Convert Julian Day to date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
        4th ed., Duffet-Smith and Zwart, 2011.

    Parameters
    ----------
    jd : float
        Julian Day

    Returns
    -------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.

    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.

    day : float
        Day, may contain fractional part.

    Examples
    --------
    Convert Julian Day 2446113.75 to year, month, and day.

    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)

    """
    jd = jd + 0.5

    F, I = math.modf(jd)
    I = int(I)

    A = math.trunc((I - 1867216.25)/36524.25)

    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.)
    else:
        B = I

    C = B + 1524

    D = math.trunc((C - 122.1) / 365.25)

    E = math.trunc(365.25 * D)

    G = math.trunc((C - E) / 30.6001)

    day = C - E + F - math.trunc(30.6001 * G)

    if G < 13.5:
        month = G - 1
    else:
        month = G - 13

    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715

    return year, month, day

def assign_season(df):
    '''takes in a dataFrame possessing year and month, appends a season column.
    '''
    df['season'] = df.year - 1985
    next_ids = df.month > 5
    df.season[next_ids] += 1
    return df


def plot_season_postage_stamps(master, season_agg, ylim=(13.7, 13.3), savefig_file='../results/Anon1.pdf', ylabel='$V$'):
    '''
    Plots all the available seasons of photometry in phase-folded postage stamps.
    '''
    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(hspace=0.1, bottom=0.06, top=0.94, left=0.12, right=0.94)
    n_seasons = len(season_agg.season)

    for i in range(n_seasons):
        # get the data and best-fit angular frequency
        s = season_agg.season[i]
        ids = master.season == s
        df = master[ids]
        t = df.JD_like.values
        y = df.Vmag.values
        dy = df.Verr.values
        #this_P = season_agg.P_est1[i]
        this_P = season_agg.P_est1.median()
        phased_t = np.mod(t, this_P)/this_P

        # Fit a multiterm model
        Nterms = 4
        reg = 0.1 * np.ones(2 * Nterms + 1)
        reg[:5] = 0 # no regularization on low-order terms
        if (df.year.min() == 2006):
            #TODO: change this to something sensible
            reg = 0.3 * np.ones(2 * Nterms + 1)
            reg[:3] = 0 # no regularization on low-order terms

        modelV = LombScargle(Nterms=4, regularization=reg)
        mask = y == y # We can mask flares later on
        modelV.fit(t[mask], y[mask], dy[mask])
        tfit = np.linspace(0, this_P, 100)
        yfitV = modelV.predict(tfit, period=this_P)


        # plot the phased data
        ax = fig.add_subplot(6,4,1 + i)
        plt.plot(tfit/this_P, yfitV, alpha=0.5)
        ax.errorbar(phased_t, y, dy, fmt='.k', ecolor='gray',
                    lw=1, ms=4, capsize=1.5)

        #---R-band---

        y = df.Rmag.values
        dy = df.Rerr.values
        #this_P = season_agg.P_est1[i]
        this_P = season_agg.P_est1.median()
        phased_t = np.mod(t, this_P)/this_P

        # Fit a multiterm model
        Nterms = 4
        reg = 0.1 * np.ones(2 * Nterms + 1)
        reg[:5] = 0 # no regularization on low-order terms

        ax = fig.add_subplot(6,4,1 + i)

        modelR = LombScargle(Nterms=4, regularization=reg)
        mask = y == y # We can mask flares later on
        try:
            modelR.fit(t[mask], y[mask], dy[mask])
            tfit = np.linspace(0, this_P, 100)
            yfitR = modelR.predict(tfit, period=this_P)
            plt.plot(tfit/this_P, yfitR, alpha=0.5)
        except:
            pass
            #print('Season {} did not work for some reason'.format(s))
        # plot the phased data
        ax.errorbar(phased_t, y, dy, fmt='.k', ecolor='gray',
                    lw=1, ms=4, capsize=1.5)
        #------------


        #---B-band---

        y = df.Bmag.values
        dy = df.Berr.values
        #this_P = season_agg.P_est1[i]
        this_P = season_agg.P_est1.median()
        phased_t = np.mod(t, this_P)/this_P

        # Fit a multiterm model
        Nterms = 4
        reg = 0.1 * np.ones(2 * Nterms + 1)
        reg[:5] = 0 # no regularization on low-order terms

        ax = fig.add_subplot(6,4,1 + i)

        modelB = LombScargle(Nterms=4, regularization=reg)
        mask = y == y # We can mask flares later on
        try:
            modelB.fit(t[mask], y[mask], dy[mask])
            tfit = np.linspace(0, this_P, 100)
            yfit = modelB.predict(tfit, period=this_P)
            plt.plot(tfit/this_P, yfit, alpha=0.5)
        except:
            pass
            #print('Season {} did not work for some reason'.format(s))
        # plot the phased data
        ax.errorbar(phased_t, y, dy, fmt='.k', ecolor='gray',
                    lw=1, ms=4, capsize=1.5)
        #------------

        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_ylim(ylim)

        ax.text(0.03, 0.96, "{}".format(season_agg.years[i]),ha='left', va='top',
                transform=ax.transAxes)

        if i < 18 :
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        if i % 4 != 0:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        if i % 4 == 0:
            ax.set_ylabel(ylabel)

        if i in (18, 19, 20, 21):
            ax.set_xlabel('phase')

    plt.savefig(savefig_file, bbox_inches='tight')