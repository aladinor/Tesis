import pandas as pd
#
# path_data = '../data'
# dsd_file = 'dsd_data_77_nh_sorted_2016.csv'
#
# chunks = pd.read_csv('{}/{}'.format(path_data, dsd_file), delimiter=',', header=None, chunksize=10 ** 4,
#                      low_memory=False, na_values='None')
# i = 0
# for chunk in chunks:
#     i += 1
#     dsd_data_final = chunk.copy()
#     dsd_data_final['Date_str'] = chunk[0].astype(str)
#     dsd_data_final['Date'] = pd.to_datetime(dsd_data_final['Date_str'], format='%Y-%m-%d %H:%M:%S')
#     dsd_data_final = dsd_data_final.drop_duplicates('Date', keep='first')
#     dsd_data_final.set_index('Date', inplace=True)
#     print i

df_results = pd.DataFrame(index=pd.Index(1990, 1, 1))
print (df_results)