import torch, argparse, os, json
from tools import calc_precision_recall

def main(opts):
    ranking_result = torch.load(opts.ranking_result_path)
    cat_info = torch.load(opts.categorize_info_path)
    eval_result = calc_precision_recall(ranking_result, cat_info)
    print('Save the evaluation result, precision: {:.2f}, recall: {:.2f}'.format(eval_result['precision'], eval_result['recall']))
    torch.save(eval_result, os.path.join(opts.out_dir, 'evaluation_result.pkl'))
    with open(os.path.join(opts.out_dir, 'evaluation_result.json'), mode = 'w') as f:
        __temp = {'mean_result': eval_result['mean_result'], 'precision': eval_result['precision'], 'recall': eval_result['recall']}
        json.dump(__temp, fp = f, indent = 4)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranking_result_path', type = str, default = '/home/lzj/code/sgg/outputs/ranking_result_filted.pkl')
    parser.add_argument('--categorize_info_path', type = str, default = '/home/lzj/code/sgg/outputs/categorize_info_raw.pkl')
    parser.add_argument('--out_dir', type = str, default = '/home/lzj/code/sgg/outputs')
    opts = parser.parse_args()
    main(opts)