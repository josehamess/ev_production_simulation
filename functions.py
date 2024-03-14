import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def filter_data(data, filter_dict):
    data = data.copy()
    for key, value in filter_dict.items():
        data = data[data[key] == value]
    return data


def keep_cols(data, cols_to_keep):
    data = data.copy()
    return data[cols_to_keep]


def make_cols_numeric(data):
    data = data.copy()
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    return data


def merge_cols(data):
    data = data.copy()
    data['Vehicle Count'] = np.array([np.sum(data.iloc[i, 1:]) for i in range(len(data))])
    return data[['Date', 'Vehicle Count']]


def conv_to_annual(data):
    data = data.copy()
    data = data.iloc[[i for i in range(len(data)) if data['Date'].iloc[i][5:7] == 'Q4'], :]
    data['Date'] = data['Date'].apply(lambda x: x[0:5])
    return data


def get_change(data):
    data = data.copy()
    data['Vehicle Count'] = data['Vehicle Count'] - data['Vehicle Count'].shift(1).dropna()
    return data


def multiplier(data):
    data = data.copy()
    data['Vehicle Count'] = data['Vehicle Count'] * 1000
    return data


def get_normalised_total(data, data_name):
    data = data.copy()
    norm_totals = [data[data['name'] == data_name].iloc[i + 1, 1] - data[data['name'] == data_name].iloc[i, 1] for i in range(len(data[data['name'] == data_name]) - 1)]
    return norm_totals


def calculate_deregistered_cars(data):
    data = data.copy()
    data['Date'] = pd.to_numeric(data['Date'], errors='coerce')
    data_list = [data]
    car_cat_pairs = [[cat + ' total', cat + ' new'] for cat in ['electric', 'fossil', 'gas', 'other']]
    for cat_pair in car_cat_pairs:
        dates = list(data[data['name'] == cat_pair[1]]['Date'])
        normalised_total = get_normalised_total(data, cat_pair[0])
        new_vehicles = list(data[data['name'] == cat_pair[1]]['Vehicle Count'])
        dereg_count = [new_vehicles[i] - normalised_total[i] for i in range(len(new_vehicles))]
        data_name = [cat_pair[0].split(' ')[0] + ' old' for i in range(len(dereg_count))]
        data_list.append(pd.DataFrame({'Date':dates, 'Vehicle Count':dereg_count, 'name':data_name}))
    return pd.concat(data_list, axis=0)


def load_data(data_info_dict):

    data_list = []
    for data_name, data_info in data_info_dict.items():
        print(f'... {data_name} ...')
        data = pd.read_excel(f"UK/{data_info['file name']}", 
                             sheet_name=data_info['sheet name'], 
                             skiprows=data_info['row skip'])
        data = filter_data(data, data_info['filter info'])
        data = keep_cols(data, data_info['keep cols'])
        data = make_cols_numeric(data)
        if data_info['merge cols']:
            data = merge_cols(data)
        data = data.rename(columns=data_info['rename cols'])
        if data_info['convert date']:
            data = conv_to_annual(data)
        if data_info['change']:
            data = get_change(data)
        if data_info['multiplier']:
            data = multiplier(data)
        data['name'] = np.full((len(data)), data_name)
        data_list.append(data)

    return pd.concat(data_list, axis=0)


def conv_to_arr(var_dict):
    return np.array(list(var_dict.keys())), np.array(list(var_dict.values()))

    
def create_points(info):
    point_dict = {}
    for datapoint in info[2:]:
        if info[0] == 'fossil new':
            point_dict[datapoint[0]] = datapoint[1][0] + ((100 - info[1]) / 100) * (datapoint[1][1] - datapoint[1][0])
        else:
            point_dict[datapoint[0]] = datapoint[1][0] + (info[1] / 100) * (datapoint[1][1] - datapoint[1][0])
    return point_dict


def construct_polynomial(x, y):
    coeffs = np.polyfit(x.astype(float), y.astype(float), 5)
    polynomial = np.poly1d(coeffs)
    x_curve = np.array(range(int(x[2]), int(x[-1]) + 1, 1))
    y_curve = polynomial(x_curve)
    return x_curve, y_curve


def add_old_data(x, y, data):
    print(data)
    prev_x = np.array([float(data.iloc[0, 0]), float(data.iloc[-5, 0]), float(data.iloc[-1, 0])])
    prev_y = np.array([float(data.iloc[0, 1]), float(data.iloc[-5, 1]), float(data.iloc[-1, 1])])
    return np.append(prev_x, x), np.append(prev_y, y)


def calculate_dereg_new_ratio(df):
    df = df.copy()
    new_arr = np.zeros((8))
    old_arr = np.zeros((8))
    total_arr = np.zeros((9))
    for name in np.unique(df['name']):
        if name.split(' ')[1] == 'new':
            new_arr += np.array(df[df['name'] == name]['Vehicle Count'])
        elif name.split(' ')[1] == 'old':
            old_arr += np.array(df[df['name'] == name]['Vehicle Count'])
        elif name.split(' ')[1] == 'total':
            total_arr += np.array(df[df['name'] == name]['Vehicle Count'])
    dereg_new_ratio = old_arr / new_arr
    dereg_new_ratio_df = pd.DataFrame({'Date':list(df[df['name'] == 'electric new']['Date']), 
                                       'Vehicle Count':list(dereg_new_ratio),
                                       'name':list(np.full((8), 'dereg new ratio'))})
    total_total_df = pd.DataFrame({'Date':list(df[df['name'] == 'electric total']['Date']), 
                                       'Vehicle Count':list(total_arr),
                                       'name':list(np.full((9), 'total total'))})

    return pd.concat([df, dereg_new_ratio_df, total_total_df], axis=0)


def remove_negatives(y):
    return [val if val >= 0 else 0 for val in y]


def get_last_true_total(df):
    df = df.copy()
    last_total_dict = {}
    total_total = 0
    for name in ['electric total', 'gas total', 'other total', 'fossil total']:
        last_total_dict[name] = df[df['name'] == name]['Vehicle Count'].iloc[-1]
        total_total += df[df['name'] == name]['Vehicle Count'].iloc[-1]
    last_total_dict['total total'] = total_total
    return last_total_dict


def get_old_and_total(df_proj, last_true_total_dict):
    df_proj = df_proj.copy()
    old_dict = {'fossil old':[], 'electric old':[], 'gas old':[], 'other old':[]}
    total_dict = {'fossil total':[], 'electric total':[], 'gas total':[], 'other total':[], 'total total':[]}
    for i in range(len(df_proj)):
        total_total = 0
        for name in ['fossil', 'electric', 'gas', 'other']:
            if i == 0:
                total = last_true_total_dict[name + ' total']
                total_ratio = total / last_true_total_dict['total total']
            else:
                total = total_dict[name + ' total'][i - 1]
                total_ratio = total / total_dict['total total'][i - 1]
            total_total += total
            old = total_ratio * df_proj['all old'].iloc[i]
            new = df_proj[name + ' new'].iloc[i]
            old_dict[name + ' old'].append(old)
            total_dict[name + ' total'].append(total - old + new)
        total_dict['total total'].append(total_total)
    old_df = pd.DataFrame(old_dict)
    total_df = pd.DataFrame(total_dict)
    return pd.concat([df_proj, old_df, total_df], axis=1)


def correct_first_point(df, df_proj, name):
    vehicle_counts = df_proj[name + ' total'] / 1000000
    true_initial_count = df[df['name'] == name + ' total']['Vehicle Count'].iloc[-1]
    return [true_initial_count / 1000000] + list(vehicle_counts)[1:]


def plot_data(df, df_proj):
    fig, ax = plt.subplots(figsize=(12, 6))
    #plt.figure(figsize=(12, 6))
    ax.set_title('Total vehicles in the UK by fuel type')
    #ax.grid()
    colours = ['green', 'red', 'blue', 'orange', 'black']
    labels = ['Battery Electric Total', 'Fossil Total', 'Other Total', 'Gas Total', 'Overal Total']
    for i, name in enumerate(['electric', 'fossil', 'other', 'gas', 'total']):
        ax.plot(df_proj['Date'], correct_first_point(df, df_proj, name), color=colours[i], linestyle='--')
        ax.plot(df[df['name'] == name + ' total']['Date'].astype(int), df[df['name'] == name + ' total']['Vehicle Count'] / 1000000, label=labels[i], color=colours[i])
        #plt.scatter(x_points_dict[name], [y / 1000000 for y in y_points_dict[name]], color=colours[i], s=20)
    tick_positions = np.arange(int(df['Date'].iloc[0]), df_proj['Date'].iloc[-1], 1)  # Adjust the range and step as needed
    ax.set_xticks(tick_positions, rotation='vertical')
    ax.set_xlabel('date')
    ax.set_ylabel('total vehicles (millions)')
    ax.legend()
    st.pyplot(fig)


def get_all_old(df_proj, df):
    all_old_min = []
    for i in range(len(df_proj)):
        date = max([df_proj['Date'].iloc[i] - 15, np.min(df['Date']) + 1])
        if date > 2022:
            new_at_date = df_proj[df_proj['Date'] == date][['fossil new', 'electric new', 'gas new', 'other new']].sum().sum() * 0.75
        else:
            c1 = (df['Date'] == date)
            c2 = (df['name'].isin(['fossil new', 'electric new', 'gas new', 'other new']))
            new_at_date = np.sum(df[c1 & c2]['Vehicle Count']) * 0.75
        all_old_min.append(new_at_date)
    df_proj['all old'] = [max([np.sum(df_proj.iloc[i, [1, 2, 3, 4]]) * df_proj.iloc[i, -1], all_old_min[i]]) for i in range(len(df_proj))]
    return df_proj
