import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    parser = argparse.ArgumentParser(description='Codebook')
    parser.add_argument('--config', default=str(REPO_ROOT / 'vq_vae' / 'configs' / 'codebook.yml'))
    parser.add_argument('--gpu', type=int, default=0, help="Primary CUDA device index")
    parser.add_argument('--gpus', type=int, nargs='*', default=[0],help = "Optional list of GPU ids for DataParallel, e.g. --gpus 0 1")
    parser.add_argument('--data_path', type=str, required=False, default=None, help = "Path to the dataset file (overrides config if provided)")
    #parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--save_path', type=str, required=False, default=str(REPO_ROOT / 'vq_vae' / 'output'))
    parser.add_argument('--code_path', type=str, required=False)
    parser.add_argument(
        '--VQVAE_model_path',
        type=str,
        required=False,
        default=str(REPO_ROOT / 'vq_vae' / 'output' / 'train_codebook' / 'codebook_checkpoint_best.bin'),
    )
    parser.add_argument('--speaker_link_len', type=str, default=None)
    parser.add_argument('--skeleton_info_path', type=str, default=None)
    parser.add_argument('--step', type=str, default="1")
    parser.add_argument('--stage', type=str, default="inference")
    #args = parser.parse_args()
    args, _ = parser.parse_known_args() #to avoid conflicts with other parsers
    return args
