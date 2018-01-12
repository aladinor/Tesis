import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import math
import csv
import collections


def calc_nd(sr_event, dict_speed, dsize=.125, area=45.6 / (100 ** 2), dt=60):
    ND = 0

    for speed in sr_event.index:
        if pd.notnull(sr_event.loc[speed]):
            dspeed = dict_speed[speed]
            ND += sr_event.loc[speed] / (speed * area * dt * dsize)
    return ND


path_data = '/media/alfonso/46CE677FCE676661/ALFONSO/UNIVERSIDAD/repositorios/thesis/data/split_data'
path_results = '../results'
dsd_file = 'dsd_data_77_nh_sorted_2016_1.csv'
dsd_number = (dsd_file[dsd_file.find("a_") + 2:dsd_file.find("_n")])
savefigs = '../figs/{}'.format(dsd_number)
savepks = '../pks/{}'.format(dsd_number)

num_gotas = 100


chunks = pd.read_csv('{}/{}'.format(path_data, dsd_file), delimiter=',', header=None, chunksize=10 ** 4,
                     low_memory=False, na_values='None')

dsd_final = pd.DataFrame()

dict_idx = {i - 43: range(i, 484, 22) for i in range(44, 66)}

list_speed = {0.1: .2, .3: .2, .5: .2, .7: .2, .9: .2,
              1.2: .4, 1.6: .4, 2.0: .4, 2.4: .4, 2.8: .4, 3.2: .4,
              3.8: .8, 4.6: .8, 5.4: .8, 6.2: .8, 7.0: .8, 7.8: .8, 8.6: .8,
              9.5: 1.,
              11.: 10.}

list_size = {.125: .125, .25: .125, .375: .125,
             .5: .25, .75: .25, 1.: .25, 1.25: .25, 1.5: .25, 1.75: .25,
             2.: .5, 2.5: .5, 3.: .5, 3.5: .5, 4.: .5, 4.5: .5,
             5.: .5, 5.5: .5, 6.: .5, 6.5: .5, 7.: .5, 7.5: .5,
             8.: 10.}

list_size = collections.OrderedDict(sorted(list_size.items()))
list_speed = collections.OrderedDict(sorted(list_speed.items()))

dsd_results = pd.DataFrame()
df_drisd_total = pd.DataFrame()

for chunk in chunks:

    dsd_data_final = chunk.copy()
    dsd_data_final['Date_str'] = chunk[0].astype(str)
    dsd_data_final['Date'] = pd.to_datetime(dsd_data_final['Date_str'], format='%Y-%m-%d %H:%M:%S')
    dsd_data_final = dsd_data_final.drop_duplicates('Date', keep='first')

    dsd_data_final.set_index('Date', drop=True, inplace=True)
    dsd_data_final.drop(['Date_str'], axis=1, inplace=True)
    sr_drisd_total = dsd_data_final[9].copy()
    dsd_data_final.drop(range(43), axis=1, inplace=True)
    dsd_data_final.sort_index(inplace=True)

    dsd_data_final[dsd_data_final < 8] = 0
    #     sr_drisd_total[sr_drisd_total < 0.1] = 0
    #     dsd_data_final[sr_drisd_total == 0 ] = 0

    idx_header = pd.MultiIndex.from_product([sorted(list_size), sorted(list_speed)], names=['Size', 'Speed'])
    dsd_data_final.columns = idx_header

    df_drisd_total = pd.DataFrame(dsd_data_final.sum())
    df_drisd_total.reset_index(inplace=True)
    df_drisd_vel = df_drisd_total.pivot(index='Speed', columns='Size', values=0)

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