import pandas as pd
import numpy as np
import functions
import streamlit as st


# sliders for user to use
v1 = st.slider('Fossil fuel vehicle decrease', 0, 100, 50)
v2 = st.slider('Electric vehicle uptake', 0, 100, 50)
v3 = st.slider('Gas vehicle uptake', 0, 100, 50)
v4 = st.slider('Other vehicle uptake', 0, 100, 50)
v5 = st.slider('Vehicle disposal', 0, 100, 50)

print(v1)
# range 0 to 100 (0 minimum rate change, 100 maximum rate change)
var_dict = {
            'Fossil decrease level':['fossil new', v1, [2030, [0, 1500000]], [2035, [0, 0]], [2040, [0, 0]]], 
            'Electric car uptake':['electric new', v2, [2030, [200000, 2000000]], [2035, [0, 3000000]], [2040, [0, 3500000]]],
            'Gas car uptake':['gas new', v3, [2030, [0, 400000]], [2035, [0, 1500000]], [2040, [0, 3000000]]],
            'Other car uptake':['other new', v4, [2030, [0, 400000]], [2035, [0, 1500000]], [2040, [0, 3000000]]],
            'Car disposal level':['dereg new ratio', v5, [2030, [0.70, 1.05]], [2035, [0.70, 1.05]], [2040, [0.70, 1.05]]]
            }


# read in previously formatted data
df = pd.read_csv('formatted_data')
print(len(df))


# create new df with simulated trajectories based on users slider input and historic data
df_dict = {'Date':[x for x in range(2022, int(var_dict[list(var_dict.keys())[0]][-1][0]) + 1)]}
for key, value in var_dict.items():
    point_dict = functions.create_points(value)
    x, y = functions.conv_to_arr(point_dict)
    x, y = functions.add_old_data(x, y, df[df['name'] == value[0]])
    x, y = functions.construct_polynomial(x, y)
    y = functions.remove_negatives(y)
    df_dict[value[0]] = y
print(df_dict)
for key, value in df_dict.items():
            print(key)
            print(len(value))
df_proj = pd.DataFrame(df_dict)


# get the total old cars
df_proj = functions.get_all_old(df_proj, df)


# finally get the old and total cars for each fuel type
df_proj = functions.get_old_and_total(df_proj, functions.get_last_true_total(df))


# plot graph
functions.plot_data(df, df_proj)
