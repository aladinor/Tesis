import pandas as pd

path_data = '../data'
dsd_file = 'dsd_data_77_nh_sorted_2016.csv'

chunks = pd.read_csv('{}/{}'.format(path_data, dsd_file), delimiter=',', header=None, chunksize=10 ** 4,
                     low_memory=False, na_values='None')


dsd_final = pd.DataFrame()
dsd_results = pd.DataFrame()
i = 0
for chunk in chunks:
    dsd_data_final = chunk.copy()
    dsd_data_final['Date_str'] = chunk[0].astype(str)
    dsd_data_final['Date'] = pd.to_datetime(dsd_data_final['Date_str'], format='%Y-%m-%d %H:%M:%S')
    dsd_data_final = dsd_data_final.drop_duplicates('Date', keep='first')
    dsd_data_final.set_index('Date', inplace=True)
    dsd_data_final.drop('Date_str', inplace=True, axis=1)
    dsd_data_final['Rain_mred'] = dsd_data_final[9]
    dsd_data_final['Rain_mred_2'] = dsd_data_final[12]
    dsd_data_final['Z_mred'] = dsd_data_final[14]
    dsd_data_final.drop(range(483), axis=1, inplace=True)
    dsd_results.append(dsd_data_final)
    print dsd_results

print (dsd_results)