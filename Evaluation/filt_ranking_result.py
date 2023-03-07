import torch, argparse, os, json
from tools import filt_ranking_result

def main(opts):
    ranking_result = torch.load(opts.ranking_result_path)
    ranking_result = filt_ranking_result(ranking_result, thr = opts.thr)
    print('Save the filted ranking result with a thr')
    torch.save(ranking_result, os.path.join(opts.out_dir, 'ranking_result_filted.pkl'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranking_result_path', type = str, default = '/home/lzj/code/sgg/outputs/ranking_result.pkl')
    parser.add_argument('--thr', type = float, default = 0.1)
    parser.add_argument('--out_dir', type = str, default = '/home/lzj/code/sgg/outputs')
    opts = parser.parse_args()
    main(opts)