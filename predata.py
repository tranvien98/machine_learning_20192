import pandas as pd
import numpy as np

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('./ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings_base = ratings_base.sort_values(['user_id', 'unix_timestamp'], ascending=[True, True])
# print(ratings_base.head(150))
df_train = pd.DataFrame(columns=r_cols)
df_test = pd.DataFrame(columns=r_cols)
dem = 1
index = []
n_user = ratings_base.iloc[-1]['user_id']
for i in range(n_user+1):
    df_data = ratings_base[ratings_base['user_id'] == i]
    # n_split = 8*df_data.shape[0]//10
    # print(df_data.iloc[:-1])
    df_train = df_train.append(df_data.iloc[:-10])
    df_test = df_test.append(df_data.iloc[-10:])
df_train.to_csv('./data/train.csv', index=False, header=False)
df_test.to_csv('./data/test.csv', index=False, header=False)
