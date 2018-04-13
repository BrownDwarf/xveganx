import numpy as np
import pandas as pd
import math
import datetime as dt
from gatspy.periodic import LombScargle, LombScargleFast
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.time_series import multiterm_periodogram
from astroML.time_series import lomb_scargle
import astroML.time_series
from astropy.io import fits
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

def flat_ASAS3(ASAS3_fn):
    '''
    Returns a "flat" pandas DataFrame given a filename
    '''
    cols = ['HJD','MAG_0','MAG_1','MAG_2','MAG_3','MAG_4',
            'MER_0','MER_1','MER_2','MER_3','MER_4','GRADE','FRAME']
    as3 = pd.read_csv(ASAS3_fn, comment='#',
                      delim_whitespace=True, names=cols)
    gi = (as3.MAG_0 < 20) & (as3.GRADE != 'D')
    as3 = as3[gi]
    as3['JD_like'] = as3.HJD + 2450000.0
    as3['date_type'] = 'HJD'
    as3['Vmag'], as3['Verr']= as3.MAG_0, as3.MER_0
    as3.drop(['HJD','MAG_0', 'MAG_1', 'MAG_2', 'MAG_3', 'MAG_4', 'MER_0', 'MER_1',
       'MER_2', 'MER_3', 'MER_4', 'GRADE', 'FRAME'], axis=1, inplace=True)

    as3['source'] = 'ASAS3'
    return as3

def flat_AAVSO(AAVSO_fn):
    '''
    Returns a "flat" pandas DataFrame given a filename
    '''
    aavso = pd.read_csv(AAVSO_fn, usecols=[0,1,2,4])
    bands = ['B', 'R', 'V']
    gi = aavso.Band.isin(bands)
    aavso = aavso[gi]
    aa_new = aavso.pivot(index='JD', columns='Band')
    new_cols = [s + 'mag' for s in bands] + [s + 'err' for s in bands]
    aa_new.columns = new_cols
    aa_new = aa_new.reset_index()
    aa_new = aa_new.rename(columns={'JD':'JD_like'})
    aa_new['date_type'] = 'JD'
    aa_new['source'] = 'AAVSO'
    return aa_new

def flat_IOMC(IOMC_fn):
    '''
    Returns a "flat" pandas DataFrame given a filename
    '''
    with fits.open(IOMC_fn) as ff:
        hdu1 = ff[1]
        df_out = pd.DataFrame({'JD_like':hdu1.data['BARYTIME']+2451544.5,
                       'Vmag':hdu1.data['MAG_V'],
                       'Verr':hdu1.data['ERRMAG_V']})
        df_out['source'] = 'Integral-OMC'
        df_out['date_type'] = 'ISDC JD'
        return df_out

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


def plot_season_postage_stamps(master, season_agg, epochs, ylim=(13.7, 13.3), savefig_file='../results/test.pdf', ylabel='$V$'):
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

        #---Mark observation epochs---
        ts_ids = (np.float(s) == epochs.AdoptedSeason)
        if ts_ids.sum() > 0:
            for ei in epochs.index.values[ts_ids.values]:
                this_phase = np.mod(epochs.JD_like[ei], this_P)/this_P
                ax.vlines(this_phase, 15.3, 10.2, linestyles=epochs.linestyles[ei],
                          colors=epochs.color[ei], alpha=0.8)
                for band, model in [('V_est', modelV), ('R_est', modelR), ('B_est', modelB)]:
                    try:
                        estimated_mag = model.predict(this_phase*this_P, period=this_P).tolist()
                        epochs.set_value(ei, band, estimated_mag)
                    except:
                        pass
                        #print('{}: Band {} could not be computed'.format(epochs.Observation[ei], band))
        #-----------------------------

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


def plot_season_minimum(master, season_agg, epochs, ylim=(0, 1.1), savefig_file='../results/test_min_spot.pdf'):
    '''
    Plots all the available seasons of photometry in phase-folded postage stamps.
    '''
    fig = plt.figure(figsize=(8.5, 11))
    fig.subplots_adjust(hspace=0.1, bottom=0.06, top=0.94, left=0.12, right=0.94)
    n_seasons = len(season_agg.season)
    ylabel='Minimum $f_\mathrm{spot}$'

    for i in range(n_seasons):
        # get the data and best-fit angular frequency
        s = season_agg.season[i]
        ids = master.season == s
        df = master[ids]
        t = df.JD_like.values
        y = df.flux_rel.values
        dy = df.flux_rel.values*0.02
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

        #---Mark observation epochs---
        ts_ids = (np.float(s) == epochs.AdoptedSeason)
        if ts_ids.sum() > 0:
            for ei in epochs.index.values[ts_ids.values]:
                this_phase = np.mod(epochs.JD_like[ei], this_P)/this_P
                ax.vlines(this_phase, 0, 1, linestyles=epochs.linestyles[ei],
                          colors=epochs.color[ei], alpha=0.8)
        #-----------------------------

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


def flatten_photometry():
    '''
    Automatically generate a flat file for all targets and all photometry
    '''
    fn_df = pd.read_csv('../data/metadata/photometry_filenames.csv')

    for i in fn_df.index:
        df_list = []
        if fn_df.Grankin08_fn[i] is not np.NaN:
            gr_data = flat_grankin08('../data/Grankin_2008/'+fn_df.Grankin08_fn[i])
            df_list.append(gr_data)
        if fn_df.ASASSN_fn[i] is not np.NaN:
            ASASSN_data = flat_ASASSN('../data/ASASSN/'+fn_df.ASASSN_fn[i])
            df_list.append(ASASSN_data)
        if fn_df.ASAS3_fn[i] is not np.NaN:
            as3 = flat_ASAS3('../data/ASAS3/'+fn_df.ASAS3_fn[i])
            df_list.append(as3)
        master = pd.concat(df_list, join='outer', ignore_index=True, axis=0)
        master['year'], master['month'], master['day'] = jd_to_date(master.JD_like.values)
        master = assign_season(master)
        col_order = ['JD_like', 'year', 'month', 'day', 'season',
                    'Vmag', 'Verr', 'Bmag', 'Berr', 'Rmag', 'Rerr', 'Umag', 'Uerr',
                     'source', 'date_type']
        master = master[col_order]
        master.Verr[master.Vmag != master.Vmag] = np.NaN
        master.Rerr[master.Rmag != master.Rmag] = np.NaN
        master.Berr[master.Bmag != master.Bmag] = np.NaN
        master.Uerr[master.Umag != master.Umag] = np.NaN
        master.to_csv('../data/flat_photometry/'+fn_df.master_fn[i], index=False)

    return 0


def master_photometry():
    '''
    Return a DataFrame with all photometry of all objects from all sources
    '''
    fn_df = pd.read_csv('../data/metadata/photometry_filenames.csv')

    master = pd.DataFrame()
    for i in fn_df.index:
        this_master = pd.read_csv('../data/flat_photometry/'+fn_df.master_fn[i])
        this_master['object'] = fn_df.name[i]
        master = master.append(this_master, ignore_index=True)

    #Add a row for flux
    master['flux_rel'] = 10**(-1.0*master.Vmag/2.5)
    master['flux_rel'] = master['flux_rel'] /np.percentile(master.V_flux_rel, [99])

    return master


def seasonal_aggregation(master, target_name):
    '''
    Return a seasonally aggregated stats, given a target name and master df
    '''
    master = master[master.object == target_name].reset_index(drop=True)
    gb = master.groupby('season')
    df_season = pd.DataFrame()

    for band in 'UBVR':
        df_season['N_'+band] = gb[band+'mag'].count()

    df_season['JD_min'] = gb.JD_like.min()
    df_season['JD_max'] = gb.JD_like.max()
    df_season['length'] = np.ceil(df_season.JD_max-df_season.JD_min)
    df_season['years'] = ''

    for i in range(len(df_season)):
        # get the data and best-fit angular frequency
        s = df_season.index[i]
        ids = master.season == s
        df = master[ids]
        val_out= "{}/{}-{}/{}".format(df.month[df.JD_like.argmin()],
                                      df.year.min(),
                                      df.month[df.JD_like.argmax()],
                                      df.year.max())
        df_season.set_value(s, 'years', val_out)

    g08_period = pd.read_csv('../data/metadata/Grankin08_physical.csv', usecols=['name', 'Period'])
    period_days, = g08_period.Period[g08_period.name == target_name].values
    df_season['P_est1'] = period_days
    df_season['P_err1'] = 0.1
    df_season.reset_index(inplace=True)

    return df_season

def run_periodograms(light_curve, P_range=[0.1, 10], samples=10000):
    '''Returns periodograms for hardcoded subset of K2 Cycle 2 lightcurve'''
    x = light_curve.time.values
    y = light_curve.flux.values
    yerr = light_curve.err.values

    periods = np.linspace(P_range[0], P_range[1], samples)

    omega = 2.00*np.pi/periods

    P_M = multiterm_periodogram(x, y, yerr, omega)
    P_LS = lomb_scargle(x, y, yerr, omega)
    return (periods, P_M, P_LS)

def top_N_periods(periods, lomb_scargle_power, n=5):
    '''Returns the top N Lomb-Scargle periods, given a vector of the periods and values'''

    # Get all the local maxima
    all_max_i = argrelmax(lomb_scargle_power)
    max_LS = lomb_scargle_power[all_max_i]
    max_periods = periods[all_max_i]

    # Sort by the Lomb-Scale power
    sort_i = np.argsort(max_LS)

    # Only keep the top N periods
    top_N_LS = max_LS[sort_i][::-1][0:n]
    top_N_pers = max_periods[sort_i][::-1][0:n]

    return top_N_pers, top_N_LS

def plot_LC_and_periodograms(lc, periods, P_M, P_LS):
    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.plot(lc.time, lc.flux, '.')
    plt.subplot(122)
    plt.step(periods, P_M, label='Multi-term periodogram')
    plt.step(periods, P_LS, label='Lomb Scargle')
    plt.legend()