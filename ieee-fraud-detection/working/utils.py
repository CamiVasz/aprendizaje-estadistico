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
                   'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2',
                   'P_email1', 'P_email2', 'R_email1', 'R_email2',
                    '_Weekdays', '_Hours', '_Days' ,
    ])
    
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


def preprocessing(Xf, yf, detect_outliers = False, convert_DT = False,
            create_features_props_over_cats = False, group_cat_prop=True):
    '''
    This function receives a complete or incomplete train transaction df
    alog with its respective labels.
    And returns a preprocessed DataFrame with:
        - No NaN values
        - Categorical variables coded
        - Missing indicator for NaN values
        - No outliers (if indicated)
    Inputs:
        Xf, yf (pandas dataframe):
            That represent data
        create_proportion_over_cats (bool)
            defines if these new features are added
        group_cat_prop (bool)
            group some categorical features with less than some fixed propotion

    Outputs:        
        X, y (pandas dataframe)
    '''
    X = Xf
    yd = yf

    ## Mandatory transofmations

    #  Separate mails in different cols
    # [TODO] fix this
    if 'P_emaildomain' in X:
        X[['P_email1', 'P_email2']] =  names_and_domains(X['P_emaildomain'])
        X.drop('P_emaildomain', axis=1, inplace=True)

    if 'R_emaildomain' in X:
        X[['R_email1', 'R_email2']] =  names_and_domains(X['R_emaildomain'])
        X.drop('R_emaildomain', axis=1, inplace=True)

    # Apply logratim to transaction
    X['LogTransactionAmt'] = np.log(X['TransactionAmt'])


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

    ## Outlier detection
    if detect_outliers:
        Xd, yd = outlier_detection(Xd, yd)

    ## Create_Delta time
    if convert_DT:
        Xd = convert_delta_time(Xd)

    ##  Create proportions
    if create_features_props_over_cats:
        from itertools import product
        # Define the new columns to be transformed
        cols = ['TransactionAmt']

        ests = ['mean', 'std'] # or ['median', 'std']

        cats = ['card1', 'card2', 'card4', 'card6']
        cats = [*cats, 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'] # Nan propagation?
        cats = [*cats, 'addr1', 'addr2']

        # Perform transformation
        for col, cat, est in product(cols, cats, ests):
            # if col, cat in Xd?
            Xd[f'{col}_to_{est}_{cat}'] = prop_col_over_est_category(Xd, col, cat, est)

    ## Group categories that have less than 1% or values
    if group_cat_prop:
        prop = 0.05
        cat, __ = get_categorical_from_df(Xd)
        categorical_vars = Xd.columns[cat]
        for cat in categorical_vars:
            Xd.loc[:, cat] = group_small_cats_inplace(Xd[cat], prop=prop)

    return Xd, yd



def prop_col_over_est_category(X, col, cat, est='mean'):
    '''
    Retorna una columna de tamaño X[col], de tal manera
    que sus componentes son
     X[col] / (estimador de X[col] en la categoria utilizada en X[cat].()
     
     WARNINGS:
         Usar est = 'std' puede propagar NaN si solo hay un elemento en cat.
     
     Inputs:
         X:
             Pandas dataframe
        col:
            column in X
        cat:
            Column in X, category
        est:
            Transformation utilizaed un df.groupby().transform
            Can be median mean or std
    Outputs:
        S: Pandas series
            as described above
    '''
    # Transform('mean') is diff to .mean()
    # Transform returns a df of equal size,
    # .mean() groups by 
    S = X[col] / X.groupby([cat])[col].transform(est)
    return S

def names_and_domains(mails):
    df = mails.str.split('.', expand=True)
    # We are translating the middle to last column rows in ['outlook', 'com', NaN]
    # But not columns in ['outlook', 'com', 'es']
    # Fill column 2 with values in column 1
    # This is to always get last domain
    df[2].fillna(df[1], inplace=True)
    df.fillna(np.nan, inplace=True)
    return df.drop(1, axis=1)

def group_small_cats_inplace(serie, prop=0.1):
    # [TODO]
    # HAcer copia no deberia ser necesario
    # el warning que me estaba saliendo no deberia importar
    from copy import deepcopy
    serie = deepcopy(serie)
    value_counts = serie.value_counts()
    # if we have less than 3 categories this is not useful
    if len(value_counts) <  3:
        return serie
    # Check type
    type_col = serie.dtype
    if  np.issubdtype(type_col, np.float):
        import warnings
        warnings.warn("Float type is unexpected in category grouping. Please cast\
                        before to a object type or int")
        
    # Filter the grouped cats
    min_required = prop * len(serie)
    non_passing = value_counts[value_counts < min_required].index
    # If is object input others else input -1
    if type_col == 'O':
        # Column type object
        serie.loc[ serie.isin(non_passing)] = ''
    else:
        # If not, please be int, and please not be using -1
        serie.loc[ serie.isin(non_passing)] = -1
    return serie

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
    date = df_trans['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df_trans['_Weekdays'] = date.dt.dayofweek
    df_trans['_Hours'] = date.dt.hour
    df_trans['_Days'] = date.dt.day
    return df_trans
