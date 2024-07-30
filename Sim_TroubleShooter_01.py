import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, skewnorm
import statsmodels.api as sm

global N_Sims
N_Sims = 100

global alpha
alpha = 0.05

# Function to get random number from normal distribution
def get_random_from_distribution(the_mean, the_STD, the_max, the_min):
    X = np.random.normal(the_mean, the_STD)
    return max(min(X, the_max), the_min)

# Function to build a list of birds and return as a DataFrame
def build_bird_list(n_birds, group, b_mean, b_std, b_min, b_max, extra_mean, extra_std=1):
    bird_data = []
    for _ in range(n_birds):
        body_size = get_random_from_distribution(b_mean, b_std, b_max, b_min)
        WL = np.round(body_size * 0.156 + 151.09, 0)
        extra = get_random_from_distribution(extra_mean, extra_std, extra_mean + 2.0, extra_mean - 2.0)
        bird_data.append({'group_ID': group, 'body_size': body_size, 'WL': WL, 'Extra': extra})
    return pd.DataFrame(bird_data)

# Function to perform statistical tests and reporting using GLM
def do_Stats(chick_feeds_df):
    chick_feeds_df['MOM_Del_Size'] = pd.to_numeric(chick_feeds_df['MOM_Del_Size'], errors='coerce')
    chick_feeds_df['WL'] = pd.to_numeric(chick_feeds_df['WL'], errors='coerce')
    chick_feeds_df['study_group'] = pd.Categorical(chick_feeds_df['study_group']).codes
    X = sm.add_constant(chick_feeds_df[['study_group', 'WL']])
    y = chick_feeds_df['MOM_Del_Size']
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    results = model.fit()
    return results.pvalues['study_group']

# Function to simulate the MOM process
def MOM_simulation():
    global N_Birds_in_Group, N_Groups, body_mean, body_STD, Trip_per_bird
    global load_mean, load_STD, Err_mean, Err_STD, max_MOM_Err_stds
    global Delivery_mean, Delivery_STD, Group_extras_Pct, Group_extras

    N_Birds_in_Group = 15
    N_Groups = 2
    body_mean = 45
    body_STD = 1
    body_min = 40
    body_max = 55
    Trip_per_bird = 10
    load_mean = 9.5
    load_STD = 2
    load_min = 4
    load_max = 15
    Err_mean = 0.53
    Err_STD = 1.71
    Err_max = 6
    Err_min = -6
    max_MOM_Err_std = 4
    Delivery_mean = 8.67
    Delivery_STD = 2.94
    Group_extras_Pct = .2
    Group_extras = [0, 2]

    bird_dfs = []
    chick_feeds_data = []
    
    for group in range(1, N_Groups + 1):
        group_name = f'Group_{group}'
        bird_df = build_bird_list(N_Birds_in_Group, group_name, body_mean, body_STD, body_min, body_max, 2.0, 1)
        bird_dfs.append(bird_df)
        
        for index, bird in bird_df.iterrows():
            for _ in range(Trip_per_bird):
                load_size = get_random_from_distribution(load_mean, load_STD, load_max, load_min)
                MOM_Arr_err = get_random_from_distribution(Err_mean, Err_STD, Err_max, Err_min)
                Bird_Arr_Size = bird['body_size'] + load_size
                MOM_Arr_Size = Bird_Arr_Size + MOM_Arr_err
                
                Del_size = get_random_from_distribution(Delivery_mean, Delivery_STD, Delivery_mean + 2 * Delivery_STD, Delivery_mean - 2 * Delivery_STD)
                MOM_del_err = get_random_from_distribution(Err_mean, Err_STD, Err_max, Err_min)
                
                Bird_Depart_Size = MOM_Arr_Size - (Del_size + bird['Extra'])
                if Bird_Depart_Size <= Bird_Arr_Size:
                    Del_size = load_size - 0.1
                    Bird_Depart_Size = MOM_Arr_Size - (Del_size + bird['Extra'])
                
                MOM_Dep_Size = Bird_Depart_Size + MOM_del_err
                MOM_Del_Size = MOM_Arr_Size - MOM_Dep_Size

                chick_feeds_data.append({
                    'study_group': bird['group_ID'],
                    'WL': bird['WL'],
                    'body_size_of_bird': np.round(bird['body_size'], 1),
                    'load_size': np.round(load_size, 1),
                    'Bird_Arr_Size': np.round(Bird_Arr_Size, 1),
                    'MOM_Arr_Size': np.round(MOM_Arr_Size, 1),
                    'Del_Size': np.round(Del_size, 1),
                    'MOM_Dep_Size': np.round(MOM_Dep_Size, 1),
                    'Bird_Depart_Size': np.round(Bird_Depart_Size, 1),
                    'MOM_Del_Size': np.round(MOM_Del_Size, 1)
                })
    
    chick_feeds_df = pd.DataFrame(chick_feeds_data)
    chick_feeds_df = chick_feeds_df[['study_group', 'WL', 'body_size_of_bird', 'load_size', 'Bird_Arr_Size', 'MOM_Arr_Size', 'Del_Size', 'MOM_Dep_Size', 'Bird_Depart_Size', 'MOM_Del_Size']]
    pValue = do_Stats(chick_feeds_df)
    return pValue

results_data = []
for i in range(N_Sims):
    p_value = round(MOM_simulation(), 2)
    results_data.append({'Group Extras': Group_extras, 'P': p_value})

results_df = pd.DataFrame(results_data)
print(results_df)
count_p_le_0_05 = (results_df['P'] <= alpha).sum()
print(f"Number iterations below {alpha}: {count_p_le_0_05} out of: {len(results_df['P'])}")
