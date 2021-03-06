import seaborn as sns
import matplotlib.pyplot as plt

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