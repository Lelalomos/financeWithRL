import pandas as pd
import os

def evaluate_report(df_eval, interpret):
    os.makedirs(os.path.join(os.getcwd(),'report'),exist_ok=True)

    df_eval['result'] = ((df_eval['predictions'] > 0) & (df_eval['actuals'] > 0)) | ((df_eval['predictions'] < 0) & (df_eval['actuals'] < 0))
    df_eval['std_couple'] = df_eval[['predictions', 'actuals']].std(axis=1)

    tic_interpret = interpret[['tic_label','tic_str']]
    tic_interpret = tic_interpret.drop_duplicates(subset=['tic_label','tic_str'])
    tic_interpret = tic_interpret.rename(columns = {"tic_label":"tickle"})

    df_eval = df_eval.merge(tic_interpret,how='left', on=['tickle'])

    list_tic = df_eval['tic_str'].unique()
    dict_eval = {
        "tic":[],
        "max_std":[],
        "min_std":[],
        "mean_std":[],
        "acc":[]
    }

    for tic in list_tic:
        temp = df_eval[df_eval['tic_str'] == tic]
        dict_eval["tic"].append(tic)
            
        temp['result'] = ((temp['predictions'] > 0) & (temp['actuals'] > 0)) | ((temp['predictions'] < 0) & (temp['actuals'] < 0))
        temp['std_couple'] = temp[['predictions', 'actuals']].std(axis=1)

        dict_eval['acc'].append(len(temp[temp['result'] == True])/len(temp.index))
        dict_eval['max_std'].append(temp['std_couple'].max())
        dict_eval['min_std'].append(temp['std_couple'].min())
        dict_eval['mean_std'].append(temp['std_couple'].mean())

    df_overview = pd.DataFrame(dict_eval)
    group_interpret = interpret[['group_str','tic_str']]
    group_interpret = group_interpret.drop_duplicates(subset=['group_str','tic_str'])
    group_interpret = group_interpret.rename(columns={"tic_str":"tic"})
    df_overview = df_overview.merge(group_interpret, how="left", on=['tic'])
    df_overview.to_excel(os.path.join(os.getcwd(),'report','overview_acc.xlsx'))


    list_group = df_overview['group_str'].unique()
    dict_group = {"group":[],"acc_mean":[],"acc_min":[],"acc_max":[]}
    for group in list_group:
        temp = df_overview[df_overview['group_str'] == group]
        dict_group['group'].append(group)
        dict_group['acc_mean'].append(temp['acc'].mean())
        dict_group['acc_min'].append(temp['acc'].min())
        dict_group['acc_max'].append(temp['acc'].max())

    df_group = pd.DataFrame(dict_group)
    df_group.to_excel(os.path.join(os.getcwd(),'report','overoview_accpergroup.xlsx'))


    print("max-std:",df_eval['std_couple'].max())
    print("min-std:",df_eval['std_couple'].min())
    print("mean-std:",df_eval['std_couple'].mean())
    print("accuracy:",len(df_eval[df_eval['result'] == True])/len(df_eval.index))