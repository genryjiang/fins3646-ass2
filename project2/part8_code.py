"""Question 8 code for answering questions

"""


import config as cfg
import zid_project2_main as main

tickers = cfg.TICKERS
start = '2000-12-29'
end = '2021-08-31'
cha_name = 'vol'
ret_freq_use = ['Daily',]
q = 3
DM_Ret_dict = main.portfolio_main(tickers, start, end, cha_name, ret_freq_use, q)[0]
Vol_Ret_mrg_df = main.portfolio_main(tickers, start, end, cha_name, ret_freq_use, q)[1]
EW_LS_pf_df = main.portfolio_main(tickers, start, end, cha_name, ret_freq_use, q)[2]

# Question 1
daily_return = DM_Ret_dict['Daily']
avg_ret_2008 = main.get_avg(daily_return, 2008)
min_index = avg_ret_2008.idxmin()
# print(min_index)

# Question 2
min_value = avg_ret_2008.min()
# print(round(min_value, 4))

# Question 3
monthly_return = DM_Ret_dict['Monthly']
avg_2019 = main.get_avg(monthly_return, 2019)
max_index = avg_2019.idxmax()
# print(max_index)

# Question 4
max_value = avg_2019.max()
# print(round(max_value, 4))

# Question 5
avg_2010 = main.get_avg(Vol_Ret_mrg_df, 2010)
tsla_vol = avg_2010['tsla_vol']
# print(round(tsla_vol, 4))

# Question 6
avg_vol_2008 = main.get_avg(Vol_Ret_mrg_df, 2008)
avg_vol_2018 = main.get_avg(Vol_Ret_mrg_df, 2018)
v_vol_2008 = avg_vol_2008['v_vol']
v_vol_2018 = avg_vol_2018['v_vol']
ratio = v_vol_2008/v_vol_2018
# print(round(ratio, 1))

# Question 7
vol_2010 = Vol_Ret_mrg_df[Vol_Ret_mrg_df.index.year == 2010]
# print(vol_2010[['tsla', 'tsla_vol']])

# Question 8
# print(EW_LS_pf_df)

# Question 9
avg_ewp_2019 = main.get_avg(EW_LS_pf_df, 2019)
ewp_rank_1 = avg_ewp_2019['ewp_rank_1']
# print(round(ewp_rank_1, 4))

# Question 10
ls = EW_LS_pf_df['ls']
cum_ret = main.get_cumulative_ret(ls).iloc[-1]
print(round(cum_ret, 4))