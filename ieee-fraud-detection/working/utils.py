import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import datetime

def read_train_transaction(nrows = 30000,folder_path = None, 
                    undersampling = False, RandomState=10):
    '''
    Read the data files
    Input:
        folder_path (string): 
            Where train_transaction is located.
        nrows (int):
            The number of rows to output.
        undersampling (bool):
            True if undersampling is required
        RandomState (int):
            Random state to undersampling generator

    Returns:
        X (pd.DataFrame): 
            input_features
        y (output): 
            Output feature

    
    '''
    def get_X_y(train_trans):
        X = train_trans.drop('isFraud', axis=1)
        y = train_trans['isFraud']
        return X, y

    def undersample_keep_ind(rus, X, y):
        X_r, y_r = rus.fit_resample(X, y)
        X_r.index = X.index[rus.sample_indices_]
        y_r.index = y.index[rus.sample_indices_]
        return X_r, y_r

    if not folder_path:
        folder_path = ''
        
    file_path = folder_path + 'train_transaction.csv'

    if not undersampling:
        train_transaction = pd.read_csv( file_path, nrows=nrows, index_col=0)
        X, y = get_X_y(train_transaction)
        return X, y

    else:
        from imblearn.under_sampling import RandomUnderSampler
        
        # Data will be readed from chuncks to ensure not to overload memory 
        # and return requested size of data
        rus = RandomUnderSampler(random_state=RandomState)
        chunk_size = nrows
        train_transaction_iterator = pd.read_csv( file_path, chunksize=chunk_size, index_col=0)
        # Acc_table to fill with data, the expected number of rows will be nrows
        X_acc, y_acc = pd.DataFrame(), pd.Series(dtype='int64')
        for table in train_transaction_iterator:
            if len(X_acc)  >= nrows:
                break
            # get X, y
            X, y = get_X_y(table)
            
            # Undersample X, y, keep indexes
            X_resampled, y_resampled = undersample_keep_ind(rus, X, y)
            
            # Accumulate
            X_acc = pd.concat([X_acc, X_resampled])
            y_acc = pd.concat([y_acc, y_resampled])
        # Truncate if neccesary
        X_acc = X_acc[: nrows]
        y_acc = y_acc[: nrows]
        return X_acc, y_acc

def get_categorical_from_df(X):
    '''
    This function receives a complete or incomplete train transaction df
    And return its categorical values.
    Inputs:
        X (pandas dataframe)
    Outputs:
        
        cat_X:
            The indexes of categorical values in columns
        cont_X:
            The other indexes 
    '''
    categorical_cols = set(['ProductCD',
                   *[f'card{i}' for i in range(1,7)],
                   *[f'M{i}' for i in range(1,10)],
                   'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2'])
    
    cat_X = []
    cont_X = []
    for i, col in enumerate(X):
        if col in categorical_cols:
            cat_X.append(i)
        else:
            cont_X.append(i)
            
    return cat_X, cont_X

def plot_counts_and_proportion(table, x, hue, n_most_common=4, savefig=False,figname=""):
    '''
    Function that plots
    inputs:
        df (pandas dataframe) 
            df contains x and hue categorical values in columns.
        x (string)
            column of df that wants to be explored
        hue (string)
            column of df which effect want to be seen
        normalize_over_x (bool)
            if true, normalization will be done over the subgroups created by x
            else, normalization will be done over the subgroups created by hue
    '''
    y = 'prop'
    
    # Keep just elements with most apparitions
    most_common_x = table[x].value_counts().index[:n_most_common]
    df = table[table[x].isin(most_common_x)]
    
    f, axes = plt.subplots(1, 2)
    sns.set()
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=12)
    # Plot 2 first to set order in xticks
    props = df[hue].groupby(df[x]).value_counts(normalize=True).unstack()
    props.plot(kind='bar', stacked='True', ax=axes[1], title=f'Proportions of {hue}')
    plt.xlabel(x, fontsize=15)
    
    # Plot 1
    order = list(props.index)
    chart = sns.countplot(x=x, data=df, ax=axes[0], order=order)
    chart.set_xlabel(x, fontsize=15)
    chart.set_ylabel('Count', fontsize=15)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    
    
    if savefig:
        if not figname:
            figname = f"./imgs/created/relation_{x}.pdf"
        f.savefig(figname, bbox_inches='tight')
    return f

def preprocessing(Xf, yf, detect_outliers = False, convert_DT = False):
    '''
    This function receives a complete or incomplete train transaction df
    alog with its respective labels.
    And returns a preprocessed DataFrame with:
        - No NaN values
        - Categorical variables coded
        - Missing indicator for NaN values
        - No outliers (if indicated)
    Inputs:
        X, y (pandas dataframe)
    Outputs:        
        X, y (pandas dataframe)
    '''
    X = Xf
    yd = yf
    # Extracting categorical variables
    cat, con = get_categorical_from_df(X)
    categorical_vars = X.columns[cat]
    # Coding categorical variables (ignoring NaN)
    df_raw = X[categorical_vars]
    df_temp = df_raw.astype("category").apply(lambda x: x.cat.codes)
    X[categorical_vars] = df_temp.where(~df_raw.isna(), df_raw)
    continuous_vars = X.columns.difference(categorical_vars)
    # Inputing nan
    # Mean for the continous variables 
    # Mode for the categorical variables
    cat_nan = []
    for i in categorical_vars:
        if any(pd.isnull(X[i])):
            cat_nan.append(i + 'isnan')
    con_nan = []
    for i in continuous_vars:
        if any(np.isnan(X[i])):
            con_nan.append(i + 'isnan')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator = True)
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent', add_indicator = True)
    X_cat = imp_mode.fit_transform(X.loc[:,categorical_vars])
    X_cont = imp_mean.fit_transform(X.loc[:,continuous_vars])
    X_nan_cat = pd.DataFrame(X_cat[:, len(categorical_vars):], columns = cat_nan, index = X.index)
    X_nan_cont = pd.DataFrame(X_cont[:, len(continuous_vars):], columns = con_nan, index = X.index)
    X.loc[:,categorical_vars] = X_cat[:, 0:len(categorical_vars)]
    X.loc[:,continuous_vars] = X_cont[:, 0:len(continuous_vars)]
    Xd = pd.concat([X, X_nan_cat, X_nan_cont], axis = 1)
    if detect_outliers:
        Xd, yd = outlier_detection(Xd, yd)
    if convert_DT:
        Xd = convert_delta_time(Xd)
    return Xd, yd

def outlier_detection(X, y):
    clf = IsolationForest(n_estimators=20)
    clf.fit(X)  # fit the added trees  
    ind = clf.predict(X) > 0
    X_no = X.loc[ind]
    y_no = y.loc[ind]
    return X_no, y_no

def convert_delta_time(X):
    df_trans = X
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df_trans["Date"] = df_trans['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df_trans['_Weekdays'] = df_trans['Date'].dt.dayofweek
    df_trans['_Hours'] = df_trans['Date'].dt.hour
    df_trans['_Days'] = df_trans['Date'].dt.day
    return df_trans