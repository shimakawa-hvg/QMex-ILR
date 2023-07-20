# %%
import pandas as pd
import chemical_category as cc

df_qmex = pd.read_csv('./data/dataset/bench_dGs_with_qmex_predict.csv')
df_cate = pd.read_csv('./data/dataset/bench_dGs_with_category.csv')

df_feature = df_qmex.loc[:,'mw':]
df_category = df_cate.loc[:,'polar':]

df_interaction, df_pf = cc.make_interaction_term(df_feature, df_category, degree=1, drop_rate=0.1, exclude_low_degree=False, is_new=False)

# %%
