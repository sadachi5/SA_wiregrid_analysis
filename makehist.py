import os, sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd;

from utils import colors;

def main(database, baseselect=[''],outfile='aho.png',verbose=0) :
    #import pickle5;
    #with open(database, 'rb') as f :
    #    df = pickle5.load(f);
    #    pass;
    df = pd.read_pickle(database);
    print('pandas column names = {}'.format(df.columns.values));
    if verbose>0 :
        pd.set_option('display.max_columns', 20)
        print('---- Pandas head in {} -----------'.format(database));
        print(df.head());
        print('----------------------------------');
        pd.set_option('display.max_columns', 5)
        pass;
    #Example of hist()
    #plt.hist(x, bins=36*2, range=(0.,2.*np.pi), normed=False, weights=None,
    #         cumulative=False, bottom=None, histtype='bar',
    #         align='mid', orientation='vertical', rwidth=None,
    #         log=False, color=kBlue, alpha=0.3, label='All', stacked=False,
    #         hold=None, data=None);
 
    fig, axs = plt.subplots(1,1);
    fig.tight_layout(rect=[0,0,1,1]);
    plt.subplots_adjust(wspace=1, hspace=1, left=0.15, right=0.95,bottom=0.15, top=0.95)
 
    ihist = 0;
 
    baselabel = 'All';
    if len(baseselect)>1 :  baselabel = baseselect[1];
    elif len(baseselect[0])>0 : baselabel = baseselect[0];
    baselabel = baseselect[1] if len(baseselect)>1 else 'All';
    axs.hist(df.query(baseselect[0])['wireangle0'], bins=36*2, range=(0.,2.*np.pi), histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[ihist], alpha=0.3, label=baselabel, stacked=False);
    ihist +=1;
 
    selections = [\
        ["bolo_type=='T' & pixel_type=='U'", 'UT'],\
        ["bolo_type=='B' & pixel_type=='U'", 'UB'],\
        ["bolo_type=='T' & pixel_type=='Q'", 'QT'],\
        ["bolo_type=='B' & pixel_type=='Q'", 'QB'],\
        ];
    data_selects = [];
    labels = [];
    ndata = len(selections);

    for i, selectinfo in enumerate(selections) :
        selection   = selectinfo[0] + ('' if len(baseselect[0])==0 else ('&' + baseselect[0]));
        selectlabel = selectinfo[1] if len(selectinfo)>1 else selection.strip().replace("'",'').replace('==','=').replace('_type','');
        labels.append(selectlabel);
        df_select = df.query(selection);
        print('selection = {}'.format(selection));
        print('    # of bolos = {}'.format(len(df_select)));
        data_selects.append(df_select['wireangle0']);
        pass;


    axs.hist(data_selects, bins=36*2, range=(0.,2.*np.pi), histtype='stepfilled',
             align='mid', orientation='vertical', log=False, linewidth=0.5, linestyle='-', edgecolor='k',
             color=colors[ihist:ihist+ndata], alpha=0.4, label=labels, stacked=True);
 
    axs.set_title('wireangle0');
    axs.set_xlabel(r'$\theta_{wire}(\theta=0)$',fontsize=16);
    axs.set_ylabel(r'# of bolometers',fontsize=16);
    #axs.set_xlim(-5000,5000);
    #axs.set_ylim(-5000,5000);
    axs.tick_params(labelsize=12);
    axs.grid(True);
    axs.legend(mode = 'expand',framealpha = 1,frameon = False,fontsize = 7,title='',borderaxespad=0.);
 
    fig.savefig(outfile);
 
    return 0;

if __name__=='__main__' :
    database = 'output_ver2/db/all_pandas.pkl';
    main(database, baseselect=['boloname==boloname','All'], outfile='aho_all.png',verbose=1);
    main(database, baseselect=['band==90','90GHz all'], outfile='aho_90GHz.png',verbose=1);
    main(database, baseselect=['band==150','150GHz all'], outfile='aho_150GHz.png',verbose=1);
    pass;
