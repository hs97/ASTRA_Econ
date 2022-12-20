"""
Code for self-training with weak supervision for regression analysis.
Author: Haoyu Sheng (haoyu_sheng@brown.edu)
"""
from fredapi import Fred
import pandas as pd
import statsmodels.formula.api as smf
import joblib
import numpy as np

def predict_reg(x, cutoff_date, yname, xnames):
    """
    This function generates rule predictions
    """
    x_train = x[x['index'] < cutoff_date]
    formula = yname + " ~ " + ' + '.join(xnames)
    mod = smf.ols(formula=formula, data=x_train)
    res = mod.fit()
    return res.predict(x.reset_index()[xnames])

def split_df(df, continuous=False, ffill=False, thres=2):
    '''
    This function first splits the dataframes into labels, features, and rule predictions
    It then splits the labelled data and the unlabelled data, as well as the rule predictions
    '''
    # Split dataset into labels, rule predictions, and features
    labels = df['GDP']
    x = df.loc[:, ~(df.columns.str.startswith('y_'))].drop(columns=['GDP'])
    if ffill:
        x = x.fillna(method='ffill')
    x = x.fillna(0)
    rule_preds = df.loc[:, df.columns.str.startswith('y_')]
    # Further split into labeled and unlabeled. 
    unlabeled_ind = labels.isna()
    # Fill the abstaining votes as -999
    labels[labels.isna()] = -999
    rule_preds[rule_preds.isna()] = -999
    unlabeled_labels = labels[unlabeled_ind]
    unlabeled_x = x[unlabeled_ind]
    unlabeled_rule_preds = rule_preds[unlabeled_ind]
    labeled_labels = labels[~unlabeled_ind]
    labeled_x = x[~unlabeled_ind]
    labeled_rule_preds = rule_preds[~unlabeled_ind]
    if not continuous: 
        unlabeled_labels, unlabeled_rule_preds, labeled_labels, labeled_rule_preds = to_classes(unlabeled_labels, thres=thres), to_classes(unlabeled_rule_preds, thres=thres), to_classes(labeled_labels, thres=thres), to_classes(labeled_rule_preds, thres=thres)
    return unlabeled_x, unlabeled_labels, unlabeled_rule_preds, labeled_x, labeled_labels, labeled_rule_preds

def to_classes(x, thres=2):
    """
    This function changes our output into discrete variables
    """
    x_new = x >= thres
    x_new[(x.isna()) | (x == -999)] = -1 
    return x_new.astype(int)

if __name__ == "__main__":

    reg = True
    ffill = False
    thres = 2
    include_EU = True
    # Folder extension based on cleaning assumptions
    if reg:
        ext = '_reg'
    else:
        ext = ''
    
    if ffill:
        fill_ext = "_ffill"
    else:
        fill_ext = ''
    if thres != 2 and not reg:
        thres_ext = f'_{thres}'
    else:
        thres_ext = ''
    if include_EU and reg:
        EU_ext = "_EU"
    else:
        EU_ext = ''

    fred = Fred(api_key='30adf5295a539a48e57fe367896a60e9')
    GDP = fred.get_series('GDPC1', units='pc1', frequency='q')
    u = fred.get_series('UNRATE', units='ch1', frequency='m')
    cpi = fred.get_series('CPILFESL', units='pc1', frequency='m')
    r = fred.get_series('DFF', frequency='m')
    df = pd.concat([u, cpi, r, GDP, GDP.shift(1), GDP.shift(2), GDP.shift(3), GDP.shift(4)], axis=1)
    df.columns = ['u', 'cpi', 'r', 'GDP', 'GDP_1', 'GDP_2', 'GDP_3', 'GDP_4']

    if EU_ext != '':
        GDP_EU = fred.get_series('CLVMNACSCAB1GQEU272020', units='pc1', frequency='q')
        u_EU = fred.get_series('LRHUTTTTEUM156S', units='ch1', frequency='m')
        cpi_EU = fred.get_series('CP0000EZ19M086NEST', units='pc1', frequency='m')
        r_EU = fred.get_series('ECBMLFR', frequency='m')
        df_EU = pd.concat([u_EU, cpi_EU, r_EU, GDP_EU, GDP_EU.shift(1), GDP_EU.shift(2), GDP_EU.shift(3), GDP_EU.shift(4)], axis=1)
        df_EU.columns = ['u', 'cpi', 'r', 'GDP', 'GDP_1', 'GDP_2', 'GDP_3', 'GDP_4']
        df = pd.concat([df, df_EU], axis=0)
    df.dropna(thresh=2, inplace=True)
    df.reset_index(inplace=True)
    # Summary of Economic Projection - something to include later
    sep = fred.get_series('FEDTARMDLR')
    # Okun's law: y = -1.9133 \Delta u + 3.3652
    # inverted taylor rule: y = 2(r-p) -2 - (p-2)
    df['y_okun'] = -1.9133 * df['u'] + 3.3652
    df['y_taylor'] = 2 * (df['r'] - df['cpi']) - (df['cpi'] - 2)
    # Estimate autoregression and OLS for output forecasting
    train_cutoff = '2010-01-01'
    dev_cutoff = '2018-01-01'
    lagged_vars = []
    for i in range(1, 5):
        lagged_vars += [f'GDP_{i}'] 
        df[f'y_AR{i}'] = predict_reg(df, train_cutoff, 'GDP', lagged_vars)
    df[f'y_inf'] = predict_reg(df, train_cutoff, 'GDP', ['cpi'])
    df.set_index('index', inplace=True)
    df_train = df[df.index < train_cutoff]
    df_dev = df[(df.index >= train_cutoff) & (df.index < dev_cutoff)]
    df_test = df[(df.index >= dev_cutoff)]
    if thres == 'mean':
        thres = df['GDP'].mean()
    unlabeled_x, unlabeled_labels, unlabeled_rule_preds, train_x, train_labels, train_rule_preds = split_df(df_train, continuous=reg, ffill=ffill, thres=thres)
    _, _, _, test_x, test_labels, test_rule_preds = split_df(df_test, continuous=reg, ffill=ffill, thres=thres)
    _, _, _, dev_x, dev_labels, dev_rule_preds = split_df(df_dev, reg, ffill=ffill, thres=thres)

    print("unlabeled")
    joblib.dump(np.array(unlabeled_x), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/unlabeled_x.pkl')
    joblib.dump(np.array(unlabeled_labels), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/unlabeled_labels.pkl')
    joblib.dump(np.array(unlabeled_rule_preds), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/unlabeled_rule_preds.pkl')

    print("\ntrain")
    joblib.dump(np.array(train_x), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/train_x.pkl')
    joblib.dump(np.array(train_labels), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/train_labels.pkl')
    joblib.dump(np.array(train_rule_preds), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/train_rule_preds.pkl')
    joblib.dump(np.zeros_like(train_rule_preds), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/train_exemplars.pkl')

    print("\ntest")
    joblib.dump(np.array(test_x), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/test_x.pkl')
    joblib.dump(np.array(test_labels), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/test_labels.pkl')
    joblib.dump(np.array(test_rule_preds), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/test_rule_preds.pkl')

    print("\ndev")
    joblib.dump(np.array(dev_x), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/dev_x.pkl')
    joblib.dump(np.array(dev_labels), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/dev_labels.pkl')
    joblib.dump(np.array(dev_rule_preds), f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/seed0/dev_rule_preds.pkl')

    with open(f'data/ECON{ext}{fill_ext}{thres_ext}{EU_ext}/rule_eval.txt', 'w') as f:
        if reg == False: 
            rules = test_rule_preds.columns
            for rule in rules:
                f.write(f'{rule} acc: {(test_rule_preds[rule] == test_labels).mean()}')
                f.write('\n')

        elif reg: 
            rules = test_rule_preds.columns
            for rule in rules:
                f.write(f'{rule} mse: {((test_rule_preds[rule] - test_labels)**2).mean()}')                
                f.write('\n')
