from fredapi import Fred
import pandas as pd
import statsmodels.formula.api as smf
import joblib
import numpy as np

def predict_reg(x, cutoff_date, yname, xnames):
    """
    This function generates rule predictions
    """
    x_train = x[x.index < cutoff_date]
    formula = yname + " ~ " + ' + '.join(xnames)
    mod = smf.ols(formula=formula, data=x_train)
    res = mod.fit()
    return res.predict(x[xnames])

def split_df(df, continuous=False):
    '''
    This function first splits the dataframes into labels, features, and rule predictions
    It then splits the labelled data and the unlabelled data, as well as the rule predictions
    '''
    # Split dataset into labels, rule predictions, and features
    labels = df['GDP']
    x = df.loc[:, ~(df.columns.str.startswith('y_'))].drop(columns=['GDP'])
    rule_preds = df.loc[:, df.columns.str.startswith('y_')]
    # Further split into labeled and unlabeled. 
    unlabeled_ind = labels.isna()
    unlabeled_labels = labels[unlabeled_ind]
    unlabeled_x = x[unlabeled_ind].fillna(0)
    unlabeled_rule_preds = rule_preds[unlabeled_ind] 
    labeled_labels = labels[~unlabeled_ind]
    labeled_x = x[~unlabeled_ind]
    labeled_rule_preds = rule_preds[~unlabeled_ind]
    if not continuous: 
        unlabeled_labels, unlabeled_rule_preds, labeled_labels, labeled_rule_preds = to_classes(unlabeled_labels), to_classes(unlabeled_rule_preds), to_classes(labeled_labels), to_classes(labeled_rule_preds)
    return unlabeled_x, unlabeled_labels, unlabeled_rule_preds, labeled_x, labeled_labels, labeled_rule_preds

def to_classes(x, thres=2):
    """
    This function changes our output into discrete variables
    """
    x_new = x >= thres
    x_new[x.isna()] = -1 
    return x_new.astype(int)


if __name__ == "__main__":
    fred = Fred(api_key='30adf5295a539a48e57fe367896a60e9')
    GDP = fred.get_series('GDPC1', units='pc1', frequency='q')
    u = fred.get_series('UNRATE', units='ch1', frequency='m')
    cpi = fred.get_series('CPILFESL', units='pc1', frequency='m')
    fff = fred.get_series('DFF', frequency='m')
    df = pd.concat([u, cpi, fff, GDP, GDP.shift(1), GDP.shift(2), GDP.shift(3), GDP.shift(4)], axis=1)
    df.columns = ['u', 'cpi', 'fff', 'GDP', 'GDP_1', 'GDP_2', 'GDP_3', 'GDP_4']
    df.dropna(thresh=2, inplace=True)
    # Summary of Economic Projection - something to include later
    sep = fred.get_series('FEDTARMDLR')
    # Okun's law: y = -1.9133 \Delta u + 3.3652
    # inverted taylor rule: y = 2(r-p) -2 - (p-2)
    df['y_okun'] = -1.9133 * df['u'] + 3.3652
    df['y_taylor'] = 2 * (df['fff'] - df['cpi']) - (df['cpi'] - 2)
    # Estimate autoregression and OLS for output forecasting
    train_cutoff = '2010-01-01'
    dev_cutoff = '2018-01-01'
    lagged_vars = []
    for i in range(1, 3):
        lagged_vars += [f'GDP_{i}'] 
        df[f'y_AR{i}'] = predict_reg(df, train_cutoff, 'GDP', lagged_vars)
    df[f'y_inf'] = predict_reg(df, train_cutoff, 'GDP', ['cpi'])
    df_train = df[df.index < train_cutoff]
    df_test = df[(df.index >= train_cutoff) & (df.index < dev_cutoff)]
    df_dev = df[(df.index > dev_cutoff)]

    unlabeled_x, unlabeled_labels, unlabeled_rule_preds, train_x, train_labels, train_rule_preds = split_df(df_train)
    _, _, _, test_x, test_labels, test_rule_preds = split_df(df_test)
    _, _, _, dev_x, dev_labels, dev_rule_preds = split_df(df_dev)
    unlabeled_x.fillna(0, inplace=True)
    train_x.fillna(0, inplace=True)
    test_x.fillna(0, inplace=True)
    dev_x.fillna(0, inplace=True)


    print("unlabeled")
    joblib.dump(np.array(unlabeled_x), 'data/ECON/seed0/unlabeled_x.pkl')
    joblib.dump(np.array(unlabeled_labels), 'data/ECON/seed0/unlabeled_labels.pkl')
    joblib.dump(np.array(unlabeled_rule_preds), 'data/ECON/seed0/unlabeled_rule_preds.pkl')

    print("\ntrain")
    joblib.dump(np.array(train_x), 'data/ECON/seed0/train_x.pkl')
    joblib.dump(np.array(train_labels), 'data/ECON/seed0/train_labels.pkl')
    joblib.dump(np.array(train_rule_preds), 'data/ECON/seed0/train_rule_preds.pkl')
    joblib.dump(np.zeros_like(train_rule_preds), 'data/ECON/seed0/train_exemplars.pkl')

    print("\ntest")
    joblib.dump(np.array(test_x), 'data/ECON/seed0/test_x.pkl')
    joblib.dump(np.array(test_labels), 'data/ECON/seed0/test_labels.pkl')
    joblib.dump(np.array(test_rule_preds), 'data/ECON/seed0/test_rule_preds.pkl')

    print("\ndev")
    joblib.dump(np.array(dev_x), 'data/ECON/seed0/dev_x.pkl')
    joblib.dump(np.array(dev_labels), 'data/ECON/seed0/dev_labels.pkl')
    joblib.dump(np.array(dev_rule_preds), 'data/ECON/seed0/dev_rule_preds.pkl')

    print(dev_labels)
    print(test_rule_preds)
    print(test_labels)
    for column in test_rule_preds.columns:
        print(column)
        print((test_rule_preds[column] == test_labels).mean())