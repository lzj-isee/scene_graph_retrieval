import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# generate a sub-VisualGenome dataset according to the index used in HumanAnnotation dataset

def main(annotation_results_path, triplet_information_path, image_folder, target_folder):
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
    print(l_filtered_users)
    print(len(l_filtered_users))
    print(len(df_triplet))

    '''get the image ids in this human annotation dataset'''
    query, target1, target2 = df_triplet['query_id'].tolist(), df_triplet['target_id1'].tolist(), df_triplet['target_id2'].tolist()
    image_ids = set(query + target1 + target2)
    image_names = set(map(lambda x: str(x) + '.jpg', image_ids))

    '''copy the images from VG_100K folder to create a query dataset'''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    count = 0
    for image_id in tqdm(image_ids):
        image_path = os.path.join(image_folder, str(image_id) + '.jpg')
        if os.path.exists(image_path):
            os.system('cp {} {}'.format(image_path, target_folder))
            count += 1
    print("copy %d images from %s to %s"%(count, image_folder, target_folder))
    files_in_target_folder = os.listdir(target_folder)
    if set(files_in_target_folder) != image_names:
        print("extra or lack of images, please check the images in the target folder")
    else:
        print("query images prepared")
    # '''prepare answer_cnt'''
    # answer_cnt = df_result[['id', 'triplet_id', 'answer']].pivot_table(index='triplet_id', columns='answer', aggfunc='count').fillna(0)
    # answer_cnt.columns = answer_cnt.columns.droplevel(0)
    # answer_cnt = answer_cnt.rename(columns={0:'o1', 1:'o2', 2:'both', 3:'neither'})
    # answer_cnt['o3'] = answer_cnt['both'] + answer_cnt['neither']
    # answer_cnt.head()

if __name__ == '__main__':
    image_folder = '/home/lzj/datasets/VisualGenome/VG_100K'
    target_folder = '/home/lzj/datasets/VisualGenome/query'
    annotation_results_path = '/home/lzj/datasets/VisualGenome/HumanAnnotation/anon_results.csv'
    triplet_information_path = '/home/lzj/datasets/VisualGenome/HumanAnnotation/triplets.csv'
    main(annotation_results_path, triplet_information_path, image_folder, target_folder)



