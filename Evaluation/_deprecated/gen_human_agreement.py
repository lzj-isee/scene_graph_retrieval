import os, torch, matplotlib.pyplot as plt, pandas as pd


def humman_agreement(answer_cnt, relevance_pred, df_triplet):
    ans = 0
    for i in range(len(df_triplet)):
        query_id, target_id1, target_id2 = df_triplet.iloc[i]['query_id'], df_triplet.iloc[i]['target_id1'], df_triplet.iloc[i]['target_id2']
        statistics = answer_cnt.iloc[i]
        o1, o2, o3, o4 = statistics['o1'], statistics['o2'], statistics['both'], statistics['neither']
        predictions = relevance_pred[str(query_id)]
        score1 = predictions[str(target_id1)]
        score2 = predictions[str(target_id2)]
        if score1 > score2:
            ans = ans + (o1 + o3 * 0.5) / (o1 + o2 + o3 + o4)
        elif score2 > score1:
            ans = ans + (o2 + o3 * 0.5) / (o1 + o2 + o3 + o4)
        elif score1 == score2 and score1 > 0:
            ans = ans + (0.5 * o1 + 0.5 * o2 + o3) / (o1 + o2 + o3 + o4)
        else:
            pass # score1 == score2 == 0
    ans = ans / len(df_triplet)
    return ans


def main(annotation_results_path, triplet_information_path, relevance_pred_path):
    df_result = pd.read_csv(annotation_results_path)
    df_result.index.name = 'id'
    df_result = df_result.reset_index()
    df_triplet = pd.read_csv(triplet_information_path)
    df_triplet = df_triplet.set_index('triplet_id')

    '''users and the numers of their labels'''
    df_result.groupby('user_id').count()['id'].sort_values(ascending=False)

    '''filter users with low answer count (<10)'''
    filter_count = 10
    user_count = df_result.groupby('user_id').count()['id']
    l_filtered_users = (user_count[user_count > filter_count]).index.to_list()

    '''prepare answer_cnt'''
    answer_cnt = df_result[['id', 'triplet_id', 'answer']].pivot_table(index='triplet_id', columns='answer', aggfunc='count').fillna(0)
    answer_cnt.columns = answer_cnt.columns.droplevel(0)
    answer_cnt = answer_cnt.rename(columns={0:'o1', 1:'o2', 2:'both', 3:'neither'})
    answer_cnt['o3'] = answer_cnt['both'] + answer_cnt['neither']

    # 读取ranking结果
    relevance_pred = torch.load(relevance_pred_path)
    score = humman_agreement(answer_cnt, relevance_pred, df_triplet)
    print('The human agreement score is: ')
    print(score)

if __name__ == '__main__':
    annotation_results_path = '/home/lzj/datasets/VisualGenome/HumanAnnotation/anon_results.csv'
    triplet_information_path = '/home/lzj/datasets/VisualGenome/HumanAnnotation/triplets.csv'
    relevance_pred_path = '/home/lzj/code/sgg/outputs/ranking_results.pkl'
    main(annotation_results_path, triplet_information_path, relevance_pred_path)