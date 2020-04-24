import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        folder_path = '~/Documents/Proyects/aprendizaje-estadistico/' \
                       + 'ieee-fraud-detection/input/'
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