import configargparse
import argparse
import os

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument(
        '--config',
        default=os.path.join(
            os.path.dirname(__file__),
            'DiffuseStyleGesture.yml',
        ),
    )
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_cuda', nargs='*', default=None)
    parser.add_argument('--dataset', type=str, default='BEAT')
    parser.add_argument('--name', type=str, default=None, help='Model family: MDM or DiffuseStyleGesture+')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-num-steps', type=int, default=None)
    parser.add_argument('--save-iters', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)

    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--dataset-info-path', type=str, default=None)
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--sep-people', type=str, default=None)
    parser.add_argument('--splits-data-path', type=str, default=None)

    parser.add_argument('--culture-config-path', type=str, default=None)
    parser.add_argument('--fishr-model-path', type=str, default=None)
    parser.add_argument('--adversarial-checkpoint-path', type=str, default=None)
    parser.add_argument('--adversarial-model-save-path', type=str, default=None)
    parser.add_argument('--vqvae-config-path', type=str, default=None)
    parser.add_argument('--alignment-model-checkpoint-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--test-output-dir', type=str, default=None)
    parser.add_argument('--args-path', type=str, default=None)

    parser.add_argument('--use-culture', type=str2bool, default=None)
    parser.add_argument('--use-adversarial', type=str2bool, default=None)
    parser.add_argument('--use-translated-text', type=str2bool, default=None)
    parser.add_argument('--use-language-features', type=str2bool, default=None)
    parser.add_argument('--use-translated-text-eval', type=str2bool, default=None)
    parser.add_argument('--use-language-features-eval', type=str2bool, default=None)
    parser.add_argument('--use-alignment-module', type=str2bool, default=None)
    parser.add_argument('--lambda-low-level-context', type=float, default=None)
    parser.add_argument('--lambda-high-level-context', type=float, default=None)
    parser.add_argument('--lambda-contrastive', type=float, default=None)
    parser.add_argument(
        '--use-culture-guidance-loss',
        '--use-cultural-guidance-loss',
        type=str2bool,
        default=None,
        help='If true, add a trainable culture classifier head on generated gestures.',
    )
    parser.add_argument(
        '--lambda-culture-guidance',
        '--lambda-cultural-guidance',
        '--lambda-culture',
        type=float,
        default=None,
        help='Weight for DSG+ generated-gesture culture classification loss.',
    )

    args = parser.parse_args()
    return args
