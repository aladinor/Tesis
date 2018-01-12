import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math
import csv

plt.style.use('classic')


def calc_nd(sr_event, dict_speed, dsize=.125, area=45.6 / (100 ** 2), dt=60):
    ND = 0

    for speed in sr_event.index:
        if pd.notnull(sr_event.loc[speed]):
            dspeed = dict_speed[speed]
            ND += sr_event.loc[speed] / (speed * area * dt * dsize)
    return ND


def calc_nt(sr_event, dict_speed, dsize=.125, area=45.6 / (100 ** 2), dt=60):
    NT = 0
    for speed in sr_event.index:
        if pd.notnull(sr_event.loc[speed]):
            dspeed = dict_speed[speed]
            NT += (sr_event.loc[speed] / (speed * area * dt))
    return NT


def calc_W(sr_event, dict_speed, dsize=.125, area=45.6 / (100 ** 2), dt=60):
    W = 0
    for speed in sr_event.index:

        if pd.notnull(sr_event.loc[speed]):
            dspeed = dict_speed[speed]
            W += (sr_event.loc[speed] / (speed * area * dt)) * size ** 3
    return W


def calc_R(sr_event, dict_speed, dsize=.125, area=45.6 / (100 ** 2), dt=60):
    R = 0
    for speed in sr_event.index:
        if pd.notnull(sr_event.loc[speed]):
            dspeed = dict_speed[speed]
            R += (sr_event.loc[speed] / (area * dt))*size**3.
    return R


def calc_Z(sr_event, dict_speed, dsize=.125, area=45.6 / (100 ** 2), dt=60):
    Z = 0
    for speed in sr_event.index:

        if pd.notnull(sr_event.loc[speed]):
            dspeed = dict_speed[speed]
            Z += (sr_event.loc[speed] / (speed * area * dt)) * size ** 6

    return Z


path_data = '../data'
path_results = '../results'
dsd_file = 'dsd_data_77_nh_sorted_2016.csv'
dsd_number = (dsd_file[dsd_file.find("a_") + 2:dsd_file.find("_n")])
savefigs = '../figs/{}'.format(dsd_number)
savepks = '../pks/{}'.format(dsd_number)

num_gotas = 10000

print savefigs, savepks

chunks = pd.read_csv('{}/{}'.format(path_data, dsd_file), delimiter=',', header=None, chunksize=10 ** 4,
                     low_memory=False, na_values='None')

dsd_final = pd.DataFrame()

dict_idx = {i - 43: range(i, 484, 22) for i in range(44, 66)}
list_speed = {0.10: .2, .3: .2, .5: .2, .7: .2, .9: .2,
              1.2: .4, 1.6: .4, 2.0: .4, 2.4: .4, 2.8: .4, 3.2: .4,
              3.8: .8, 4.6: .8, 5.4: .8, 6.2: .8, 7.0: .8, 7.8: .8, 8.6: .8,
              9.5: 1.,
              11.: 10.}

list_size = {.125: .125, .25: .125, .375: .125,
             .5: .25, .75: .25, 1.: .25, 1.25: .25, 1.5: .25, 1.75: .25,
             2.: .5, 2.5: .5, 3.: .5, 3.5: .5, 4.: .5, 4.5: .5,
             5.: .5, 5.5: .5, 6.: .5, 6.5: .5, 7.: .5, 7.5: .5,
             8.: 1.}

dsd_results = pd.DataFrame()

writer = csv.writer(open('{0}/{1}/revision_{2}.csv'.format(path_results, dsd_number, num_gotas), 'w'))
i = 0
for chunk in chunks:
    i += 1
    print (i)
    dsd_data_final = chunk.copy()
    dsd_data_final['Date_str'] = chunk[0].astype(str)
    dsd_data_final['Date'] = pd.to_datetime(dsd_data_final['Date_str'], format='%Y-%m-%d %H:%M:%S')
    dsd_data_final = dsd_data_final.drop_duplicates('Date', keep='first')

    dsd_data_final.set_index('Date', drop=True, inplace=True)
    dsd_data_final.drop(['Date_str'], axis=1, inplace=True)
    dsd_data_final.drop(range(43), axis=1, inplace=True)
    dsd_data_final.sort_index(inplace=True)

    idx_header = pd.MultiIndex.from_product([sorted(list_size), sorted(list_speed)], names=['Size', 'Speed'])
    dsd_data_final.columns = idx_header

    min_date = dsd_data_final.index[0]
    max_date = dsd_data_final.index[-1]

    sr_sum_drisd = dsd_data_final.sum(axis=1)
    idx_events = sr_sum_drisd[sr_sum_drisd > num_gotas].index

    delta_diam = pd.Series(sorted(list_size.values()), index=sorted(list_size)).astype(float)
    diameters = pd.Series(sorted(list_size.keys()), index=sorted(list_size)).astype(float)

    for event_ext in idx_events:
        sr_nd = pd.Series(index=sorted(list_size))
        sr_nt = pd.Series(index=sorted(list_size))
        sr_w = pd.Series(index=sorted(list_size))
        sr_R = pd.Series(index=sorted(list_size))
        sr_Z = pd.Series(index=sorted(list_size))

        for size in list_size:
            df_size = dsd_data_final.xs(size, level=0, axis=1)
            sr_event = df_size.loc[event_ext]
            dsize = list_size[size]

            nd = calc_nd(sr_event, list_speed, dsize)
            sr_nd.loc[size] = nd

            nt = calc_nt(sr_event, list_speed, dsize)
            sr_nt.loc[size] = nt

            W = calc_W(sr_event, list_speed, dsize)
            sr_w.loc[size] = W

            R = calc_R(sr_event, list_speed, dsize)
            sr_R.loc[size] = R

            Z = calc_R(sr_event, list_speed, dsize)
            sr_Z.loc[size] = Z

        # sr_nd.to_pickle(savepks + '/sr_nd_{0}.pkl'.format(event_ext))

        try:
            m0 = pd.Series(index=sorted(list_size))
            m0 = sr_nd * delta_diam
            M0 = m0.sum()
            dsd_results.loc[event_ext, 'M0'] = M0

            m1 = pd.Series(index=sorted(list_size))
            m1 = sr_nd * delta_diam * diameters
            M1 = m1.sum()
            dsd_results.loc[event_ext, 'M1'] = M1

            m2 = pd.Series(index=sorted(list_size))
            m2 = sr_nd * delta_diam * diameters ** 2
            M2 = m2.sum()
            dsd_results.loc[event_ext, 'M2'] = M2

            m3 = pd.Series(index=sorted(list_size))
            m3 = sr_nd * delta_diam * diameters ** 3
            M3 = m3.sum()
            dsd_results.loc[event_ext, 'M3'] = M3

            m4 = pd.Series(index=sorted(list_size))
            m4 = sr_nd * delta_diam * diameters ** 4
            M4 = m4.sum()
            dsd_results.loc[event_ext, 'M4'] = M4

            m5 = pd.Series(index=sorted(list_size))
            m5 = sr_nd * delta_diam * diameters ** 5
            M5 = m5.sum()
            dsd_results.loc[event_ext, 'M5'] = M5

            m6 = pd.Series(index=sorted(list_size))
            m6 = sr_nd * delta_diam * diameters ** 6
            M6 = m6.sum()
            dsd_results.loc[event_ext, 'M6'] = M6

            ## MICROFISICA DE LA PRECIPITACION POR DISDROMETRO ###

            NT_dsd = sr_nt.sum()
            NT_dsd_1 = M0
            W_dsd = (math.pi / 6.0) * 10 ** (-3) * sr_w.sum()
            R_dsd = 6*math.pi * 10 ** (-4.0) * sr_R.sum()
            Z_dsd = 10*np.log10(sr_Z.sum())
            Z_dsd_cal = 10*np.log10(M6)
            Da_dsd = M1 / M0
            De_dsd = M3 / M2
            Dm_dsd = M4 / M3

            dsd_results.loc[event_ext, 'NT_dsd'] = NT_dsd
            dsd_results.loc[event_ext, 'NT_dsd_1'] = NT_dsd_1
            dsd_results.loc[event_ext, 'W_dsd'] = W_dsd
            dsd_results.loc[event_ext, 'Z_dsd'] = Z_dsd
            dsd_results.loc[event_ext, 'Z_dsd_cal'] = Z_dsd_cal
            dsd_results.loc[event_ext, 'R_dsd'] = R_dsd
            dsd_results.loc[event_ext, 'Da_dsd'] = Da_dsd
            dsd_results.loc[event_ext, 'De_dsd'] = De_dsd
            dsd_results.loc[event_ext, 'Dm_dsd'] = Dm_dsd

            ###    MODELO MARSHALL AND PALMER 1948  ###

            mmt_nb = 1.0
            slp_mp = ((8000.0 * special.gamma(mmt_nb + 1.0)) / (M1)) ** (1.0 / (mmt_nb + 1.0))
            M_P = pd.Series(index=sorted(list_size))
            M_P = 8000.0 * np.exp(-slp_mp * diameters)


            NT_mp = 8000.0 / slp_mp
            W_mp = (math.pi / 6.0) * 10 ** (-3.) * (8000.0 * (special.gamma(3. + 1.) / (slp_mp ** (3. + 1.))))
            Z_mp = 8000.0 * (special.gamma(6. + 1.) / (slp_mp ** (6. + 1.)))

            R_mp = 6*10**(-4)*math.pi*8000.*3.778*(special.gamma(3.+0.67+1.)/(slp_mp**(3.+0.67+1.)))

            #### VELOCIDAD TERMINAL = 9.65 - 10.3*e^(-0.6D)

            K1 = 6 * 10 ** (-4) * math.pi * 8000.0
            R_mp_2 = K1 * (9.65 * (special.gamma(3. + 1) / (slp_mp ** (3. + 1.))) -
                           10.3 * (special.gamma(3. + 1) / ((slp_mp + 0.6) ** (3. + 1.))))

            #### VELOCIDAD TERMINAL = POLINOMICA

            R_mp_3 = K1 * (-0.1021*(special.gamma(3. + 1.) / (slp_mp ** (3. + 1.))) +
                           4.932*(special.gamma(3. + 2.) / (slp_mp ** (3. + 2.)))
                           - 0.9551*(special.gamma(3. + 3.) / (slp_mp ** (3. + 3.)))
                           + 0.07934*(special.gamma(3. + 4.) / (slp_mp ** (3. + 4.)))
                           - 0.002362*(special.gamma(3. + 5.) / (slp_mp ** (3. + 5.))))
            Da_mp = 8000.0 * (special.gamma(1. + 1.) / (slp_mp ** (1. + 1.))) / (8000.0 / slp_mp)
            De_mp = 8000.0 * (special.gamma(3. + 1.) / (slp_mp ** (3. + 1.))) / (
                8000.0 * (special.gamma(2. + 1.) / (slp_mp ** (2. + 1.))))
            Dm_mp = 8000.0 * (special.gamma(4. + 1.) / (slp_mp ** (4. + 1.))) / (
                8000.0 * (special.gamma(3. + 1.) / (slp_mp ** (3. + 1.))))

            dsd_results.loc[event_ext, 'slp_mp'] = slp_mp
            dsd_results.loc[event_ext, 'NT_mp'] = NT_mp
            dsd_results.loc[event_ext, 'W_mp'] = W_mp
            dsd_results.loc[event_ext, 'Z_mp'] = 10*np.log10(Z_mp)
            dsd_results.loc[event_ext, 'R_mp'] = R_mp
            dsd_results.loc[event_ext, 'R_mp_2'] = R_mp_2
            dsd_results.loc[event_ext, 'R_mp_3'] = R_mp_3
            dsd_results.loc[event_ext, 'Da_mp'] = Da_mp
            dsd_results.loc[event_ext, 'Da_mp'] = De_mp
            dsd_results.loc[event_ext, 'Da_mp'] = Dm_mp

            #     Modelo Exponencial

            m = 2.0
            n = 1.0
            slopexp = ((M1 * special.gamma(m + 1.0)) / (M2 * special.gamma(n + 1.0))) ** (1 / (m - n))
            N_0exp = M1 * (slopexp ** (n + 1.0)) / (special.gamma(n + 1.0))
            Exp = pd.Series(index=sorted(list_size))
            Exp = N_0exp * np.exp(-slopexp * diameters)

            Nt_exp = N_0exp / slopexp
            W_exp = (math.pi / 6.0) * 10 ** (-3.) * N_0exp * special.gamma(3. + 1.) / (slopexp ** (3. + 1.))
            Z_exp = N_0exp * special.gamma(6. + 1.) / (slopexp ** (6. + 1.))

            ## VELOCIDAD TERMINAL = aD^b
            R_exp = 6*10**(-4)*math.pi*N_0exp*3.778*(special.gamma(3.+0.67+1)/(slopexp**(3.+0.67+1)))

            #### VELOCIDAD TERMINAL = 9.65 - 10.3*e^(-0.6D)

            K = 6 * 10 ** (-4) * math.pi * N_0exp
            R_exp_2 = K * (9.65 * (special.gamma(3. + 1) / (slopexp ** (3. + 1.))) -
                           10.3 * (special.gamma(3. + 1) / ((slopexp + 0.6) ** (3. + 1.))))

            #### VELOCIDAD TERMINAL = POLINOMICA

            R_exp_3 = K * (-0.1021*(special.gamma(3. + 1.) / (slopexp ** (3. + 1.))) +
                           4.932*(special.gamma(3. + 2.) / (slopexp ** (3. + 2.)))
                           - 0.9551*(special.gamma(3. + 3.) / (slopexp ** (3. + 3.)))
                           + 0.07934*(special.gamma(3. + 4.) / (slopexp ** (3. + 4.)))
                           - 0.002362*(special.gamma(3. + 5.) / (slopexp ** (3. + 5.))))

            Da_exp = N_0exp * (special.gamma(1. + 1.) / (slopexp ** (1. + 1.))) / (N_0exp / slopexp)
            De_exp = N_0exp * (special.gamma(3. + 1.) / (slopexp ** (3. + 1.))) / (
                N_0exp * (special.gamma(2. + 1.) / (slopexp ** (2. + 1.))))
            Dm_exp = N_0exp * (special.gamma(4. + 1.) / (slopexp ** (4. + 1.))) / (
                N_0exp * (special.gamma(3. + 1.) / (slopexp ** (3. + 1.))))

            dsd_results.loc[event_ext, 'slp_exp'] = slopexp
            dsd_results.loc[event_ext, 'N0_exp'] = N_0exp
            dsd_results.loc[event_ext, 'NT_exp'] = Nt_exp
            dsd_results.loc[event_ext, 'W_exp'] = W_exp
            dsd_results.loc[event_ext, 'Z_exp'] = 10*np.log10(Z_exp)
            dsd_results.loc[event_ext, 'R_exp'] = R_exp
            dsd_results.loc[event_ext, 'R_exp_2'] = R_exp_2
            dsd_results.loc[event_ext, 'R_exp_3'] = R_exp_3
            dsd_results.loc[event_ext, 'Da_exp'] = Da_exp
            dsd_results.loc[event_ext, 'De_exp'] = De_exp
            dsd_results.loc[event_ext, 'Dm_exp'] = Dm_exp

            #      Modelo Gamma 246

            eta246 = M4 ** 2. / (M2 * M6)
            shape246 = ((7.0 - 11.0 * eta246) - (eta246 ** 2 + 14 * eta246 + 1.0) ** 0.5) / (2.0 * (eta246 - 1.0))
            slope246 = ((M2 / M4) * (shape246 + 3.0) * (shape246 + 4.0)) ** 0.5
            N0_246 = (M2 * slope246 ** (shape246 + 3.0)) / (special.gamma(shape246 + 3.0))
            gamma246 = pd.Series(index=sorted(list_size))
            gamma246 = N0_246 * (diameters ** shape246) * np.exp(-slope246 * diameters)
            NT_246 = N0_246 * (special.gamma(shape246 + 1) / (slope246 ** (shape246 + 1)))
            W_246 = (math.pi / 6.0) * 10 ** (-3.) * N0_246 * (special.gamma(shape246 + 4) / (slope246 ** (shape246 + 4)))
            Z_246 = N0_246 * (special.gamma(shape246 + 7) / (slope246 ** (shape246 + 7)))

            ## VELOCIDAD TERMINAL = aD^b

            R_246 = 6*10**(-4)*math.pi*N0_246*3.778*(special.gamma(3.+shape246+0.67+1)/(slope246**(3.+shape246+0.67+1)))

            ## VELOCIDAD TERMINAL = 9.65 - 10.3*e^(-0.6*D)

            K = 6 * 10 ** (-4) * math.pi * N0_246
            R_246_2 = K * (9.65 * (special.gamma(3. + shape246 + 1) / (slope246 ** (3. + shape246 + 1))) -
                    10.3 * (special.gamma(3. + shape246 + 1) / ((slope246 + 0.6) ** (3. + shape246 + 1))))

            ## VELOCIDAD TERMINAL = POLINOMICA

            R_246_3 = K * (-0.1021*(special.gamma(3. + shape246 + 1.) / (slope246 ** (3. + shape246 + 1.))) +
                           4.932*(special.gamma(3. + shape246 + 2.) / (slope246 ** (3. + shape246 + 2.)))
                           - 0.9551*(special.gamma(3. + shape246 + 3.) / (slope246 ** (3. + shape246 + 3.)))
                           + 0.0793*(special.gamma(3. + shape246 + 4.) / (slope246 ** (3. + shape246 + 4.)))
                           - 0.002362*(special.gamma(3. + shape246 + 5.) / (slope246 ** (3. + shape246 + 5.))))

            Da_246 = (N0_246 * (special.gamma(shape246 + 2.) / (slope246 ** (shape246 + 2.)))) / (
                N0_246 * (special.gamma(shape246 + 1.) / (slope246 ** (shape246 + 1.))))
            De_246 = (N0_246 * (special.gamma(shape246 + 4.) / (slope246 ** (shape246 + 4.)))) / (
                N0_246 * (special.gamma(shape246 + 3.) / (slope246 ** (shape246 + 3.))))
            Dm_246 = (N0_246 * (special.gamma(shape246 + 5.) / (slope246 ** (shape246 + 5.)))) / (
                N0_246 * (special.gamma(shape246 + 4.) / (slope246 ** (shape246 + 4.))))

            dsd_results.loc[event_ext, 'slp_246'] = slope246
            dsd_results.loc[event_ext, 'shp_246'] = shape246
            dsd_results.loc[event_ext, 'N0_246'] = N0_246
            dsd_results.loc[event_ext, 'NT_246'] = NT_246
            dsd_results.loc[event_ext, 'W_246'] = W_246
            dsd_results.loc[event_ext, 'Z_246'] = 10*np.log10(Z_246)
            dsd_results.loc[event_ext, 'R_246'] = R_246
            dsd_results.loc[event_ext, 'R_246_2'] = R_246_2
            dsd_results.loc[event_ext, 'R_246_3'] = R_246_3
            dsd_results.loc[event_ext, 'Da_246'] = Da_246
            dsd_results.loc[event_ext, 'De_246'] = De_246
            dsd_results.loc[event_ext, 'Dm_246'] = Dm_246

            dsd_results.append(dsd_results)

            # plt.figure(figsize=(12, 9))
            # plt.rc('text', fontsize=18, usetex=True)
            # ax = sr_nd[sr_nd > 1.].plot(style='x-', logy=True, grid=True, ylim=[10 ** 0, 10 ** 6],
            #                             xlim=[0, 5], legend=True, label='$ Observado $', fontsize=16)
            # M_P[M_P > 1.].plot(style='--', logy=True, label='$ Marshall - Palmer $', legend=True, ylim=[10 ** 0, 10 ** 6],
            #                    xlim=[0, 8])
            # Exp[Exp > 1.].plot(style='-.', logy=True, label='$ Exponencial $', legend=True, ylim=[10 ** 0, 10 ** 6],
            #                    xlim=[0, 8])
            #
            # gamma246.plot(style='+', color='k', logy=True, label='$Gamma$', legend=True,
            #               ylim=[10 ** 0, 10 ** 6], xlim=[0, 8])
            #
            # ax.set_xlabel('$Diametro, mm$', fontsize=16)
            # ax.set_ylabel('$N(D), \ \# \ m^{-3} mm^{-1} $', fontsize=16)
            # plt.savefig(savefigs + '/fig_{0}'.format(event_ext))
            # plt.close('all')
        except ZeroDivisionError, e:
            print (e)
            continue
        except OverflowError, e:
            print(e)
            continue

dsd_results.sort_index(inplace=True)
dsd_results.dropna(how='all', inplace=True)
xls_data = pd.ExcelWriter('{0}/{1}/dsd_results_{2}_{3}.xlsx'.format(path_results, dsd_number, dsd_number, num_gotas))
dsd_results.to_excel(xls_data)
xls_data.save()

## GRAFICO DE N0 Vs Shape

plt.figure(figsize=(8, 5))
plt.rc('text', usetex=True)
plt.yscale('log')
plt.scatter(dsd_results['shp_246'], dsd_results['N0_246'], marker='+')
plt.ylim([10 ** 3, 10 ** 8])
plt.xlim([-3, 7])
plt.xlabel('$\mu $', size=14)
plt.ylabel('$ N_{0},\  \# \ \ m^{-3} mm^{-\mu -1} $', size=14)
plt.savefig('{0}/N0_shp_{1}_{2}.png'.format(savefigs, dsd_number, str(num_gotas)))
plt.show()

### GRAFICO DE N0 Vs Slope

plt.figure(figsize=(8, 5))
plt.rc('text', usetex=True)
plt.yscale('log')
plt.scatter(dsd_results['slp_246'], dsd_results['N0_246'], marker='+')
plt.ylim([10 ** 2, 10 ** 7])
plt.xlim([0, 7])
plt.xlabel('$\Lambda, mm^{-1} $', size=14)
plt.ylabel('$ N_{0},\  \# \ \ m^{-3} mm^{-\mu -1} $', size=14)
plt.savefig('{0}/N0_slp_{1}_{2}.png'.format(savefigs, dsd_number, str(num_gotas)))
plt.show()

### GRAFICO DE shape Vs Slope

plt.figure(figsize=(8, 5))
plt.rc('text', usetex=True)
z = np.polyfit(dsd_results['shp_246'], dsd_results['slp_246'], 2)
z1 = np.polyfit(dsd_results['slp_246'], dsd_results['shp_246'], 2)
plt.scatter(dsd_results['slp_246'], dsd_results['shp_246'], marker='+')
plt.xlim([0, 6])
plt.ylim([-2.5, 4])
plt.xlabel('$\Lambda, mm^{-1} $', size=14)
plt.ylabel('$ \mu $', size=14)
plt.savefig('{0}/slp_shp_{1}_{2}.png'.format(savefigs, dsd_number, str(num_gotas)))
plt.show()

np.savetxt(
    '{0}/{1}/CG_relation_z_{2}_{3}.xlsx'.format(path_results, dsd_number, dsd_number, num_gotas), z, delimiter=",")
np.savetxt(
    '{0}/{1}/CG_relation_z1_{2}_{3}.xlsx'.format(path_results, dsd_number, dsd_number, num_gotas), z1, delimiter=",")
