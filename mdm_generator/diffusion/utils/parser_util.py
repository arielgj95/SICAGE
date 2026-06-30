from argparse import ArgumentParser
import argparse
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return args #apply_rules(args)

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    #hf_handler.get_dependencies()
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args




def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=str2bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='TensorboardPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    #group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    ######### 10 -> 50
    group.add_argument("--diffusion_steps", default=50, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=str2bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_dec',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--layers", default=10, type=int,
                       help="Number of layers.")
    group.add_argument("--heads", default=8, type=int,
                       help="Number of attention heads.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer width.")
    group.add_argument("--ffn_size", default=2048, type=int,
                       help="Transformer ffn size.")
    group.add_argument("--use_culture",default=True,type=str2bool, help = "whether using or not culture encodings")
    group.add_argument("--use_adversarial", default=False, type=str2bool, help="whether using or not culture encodings")
    group.add_argument(
        "--use_one_hot_culture",
        default=False,
        type=str2bool,
        help="If True, condition on a trainable projection of labels['culture_enc'] instead of a frozen Fishr/adversarial encoder.",
    )
    group.add_argument(
        "--culture_encoder_type",
        default="",
        choices=["", "none", "fishr", "adversarial", "one_hot", "onehot"],
        type=str,
        help="Optional explicit culture conditioning mode. Overrides --use_adversarial/--use_one_hot_culture when set.",
    )
    group.add_argument(
        "--use_native_attention",
        default=False,
        type=str2bool,
        help="If True, use PyTorch native TransformerDecoderLayer blocks for cross/self attention.",
    )
    group.add_argument(
        "--no-adain",
        dest="use_adain",
        action="store_false",
        default=True,
        help="Disable AdaIN high-level conditioning layers for ALaDiT ablations.",
    )
    group.add_argument(
        "--noalignment",
        "--no-alignment",
        dest="use_alignment",
        action="store_false",
        default=True,
        help="Disable ALaDiT low/high/contrastive/GRAM/collapse alignment losses.",
    )
    group.add_argument(
        "--nocl-layer",
        "--no-cl-layer",
        dest="use_culture_classification_layer",
        action="store_false",
        default=True,
        help="Disable the auxiliary culture-classification head and its loss.",
    )
    #group.add_argument("--cond_mask_prob", default=.1, type=float,
    #                   help="The probability of masking the condition during training."
    #                        " For classifier-free guidance learning.")
    #group.add_argument("--mask_frames", action='store_true',
    #                   help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--lambda_unmasked_pose", default=1., type=float,help="l2 loss for reconstructing unmasked motion.")
    group.add_argument("--lambda_masked_pose", default=1., type=float,help="l2 loss for reconstructing masked motion.")
    group.add_argument("--lambda_culture", default=.1, type=float,help="cross-entropy for culture loss.")
    group.add_argument("--lambda_low_level_context", default=.1, type=float, help="cosine alignment loss for low level context.")
    group.add_argument("--lambda_high_level_context", default=.1, type=float,help="cosine alignment loss for high level context.")
    group.add_argument("--lambda_contrastive", default=.1, type=float,help="contrastive loss modulation.")
    group.add_argument("--use_gram_loss", default=False, type=str2bool,
                       help="If True, use GRAM alignment and disable cosine + contrastive alignment losses.")
    group.add_argument("--lambda_gram", default=.1, type=float,
                       help="GRAM alignment loss modulation.")
    group.add_argument("--gram_eps", default=1e-6, type=float,
                       help="Numerical stability epsilon used for GRAM determinant computation.")
    group.add_argument("--lambda_diversity", default=.1, type=float,help="mmd loss for ensuring matching distribution with real motion")
    group.add_argument(
        "--lambda_collapse_reg",
        default=0.01,
        type=float,
        help="Variance regularization weight used to reduce representational collapse in alignment spaces.",
    )
    #group.add_argument("--emb_before_mask", action='store_true',
    #                   help="If true - for the cond branch - flip between mask and linear blocks (as described in Fig.2).")
    #group.add_argument("--emb_policy", default='concat',
    #                   choices=['add', 'concat'], type=str,
    #                   help="Time and aux embedings are either added or concatinated to the text embedings")
    #group.add_argument("--pos_embed_max_len", default=5000, type=int,
    #                   help="Pose embedding max length.")
    group.add_argument("--use_ema", default=True, type=str2bool, help="If True, will use EMA model averaging.")

    # Deprecated - old conditioning method on single joint - aka Guy's method:
    # group.add_argument("--target_cond", default=None,
    #                 choices=[None, 'pelvis', 'traj', HML_EE_JOINT_NAMES], type=str,
    #                 help="If not None - Target condition as an aux input.")
    # group.add_argument("--target_heading", action='store_true', help="Apply target heading condition.")

    #group.add_argument("--multi_target_cond", action='store_true',
    #                   help="If true, enable multi-target conditioning (aka Sigal's model).")
    #group.add_argument("--multi_encoder_type", default='single', choices=['single', 'multi', 'split'], type=str,
    #                   help="Specifies the encoder type to be used for the multi joint condition.")
    #group.add_argument("--target_enc_layers", default=1, type=int, help="Num target encoder layers")

    # Prefix completion model
    group.add_argument("--motion_prefix_len", default=5, type=int, help="Prefix length of motion for the autoregressive model.")
    group.add_argument("--motion_pred_len", default=20, type=int,help="Prediction length of motion for the autoregressive model.")
    group.add_argument("--audio_prefix_len", default=10, type=int, help="Prefix length of audio for the autoregressive model.")
    group.add_argument(
        "--motion_mask_prob",
        default=0.0,
        type=float,
        help="Random masking probability on noised motion frames during training. Set 0 to disable masking.",
    )
    group.add_argument(
        "--audio_mask_prob",
        default=0.0,
        type=float,
        help="Random masking probability on noised audio frames during training. Set 0 to disable masking.",
    )

    # Spatial conditioning
    #group.add_argument("--spatial_condition", default=None, type=str, choices=[None, 'traj'],
    #                   help="if not None, will use the PriorMDM method for spatial conditioning.")

    # Keyframe conditioning
    #group.add_argument("--keyframe_cond_type", default='',
    #                   choices=['', 'prefix'], type=str,
    #                   help="prefix conditioning us designed for the RL task. Default is no conditioning.")
    #group.add_argument("--keyframe_cond_prob", default=0.5, type=float,
    #                   help="Probability of a training example with keyframe conditioning.")

    #group.add_argument("--use_inpainting", action='store_true', help="")
    #group.add_argument("--use_recon_guidance", action='store_true', help="")


# TODO pay attention to sensitive information in paths. It's better to load from command line
def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    default_data_root = os.environ.get("SICAGE_DATASET_ROOT", str(REPO_ROOT / "data_root"))
    group.add_argument("--dataset", default='whole_dataset', choices=['whole_dataset'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--lmdb_path", default=os.path.join(default_data_root, "whole_dataset"), type=str,
                       help="Dataset lmdb_path")
    group.add_argument("--info_path", default=os.path.join(default_data_root, "whole_dataset_info"), type=str,
                       help="Contains speaker labels, culture_labels, subj_dep and subj_indep splits")
    group.add_argument("--metadata_path", default=os.path.join(default_data_root, "whole_dataset_metadata.pkl"), type=str,
                       help="Contains keys to access lmdb dataset and info on cultures and speakers")
    group.add_argument("--culture_config_path", default=str(REPO_ROOT / "culture_encoder/config.yml"), type=str,
                       help="Contains configuration file of culture classifier model")
    group.add_argument("--fishr_model_path", default="", type=str,
                       help="Optional override path to Fishr checkpoint (model_best.pt) used by Hierarchical_MDM.")
    group.add_argument("--adversarial_checkpoint_path", default="", type=str,
                       help="Optional override path to adversarial culture checkpoint (*.bin) used by Hierarchical_MDM.")
    group.add_argument("--adversarial_model_save_path", default="", type=str,
                       help="Optional override root path containing adversarial checkpoints.")
    group.add_argument("--vqvae_config_path", default=str(REPO_ROOT / "vq_vae/configs/codebook.yml"), type=str,
                       help="Contains configuration file of culture classifier model")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", default=str(REPO_ROOT / "mdm_runs/hierarchical_mdm_results"), required=False, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", default=True, type=str2bool,
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=5e-5, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0., type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    #group.add_argument("--eval_batch_size", default=32, type=int,
    #                   help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
    #                        "T2m precision calculation is based on fixed batch size 32.")
    #group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
    #                   help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", default=False, type=str2bool,
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=10, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    #group.add_argument("--eval_num_samples", default=1_000, type=int,
    #                   help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=500, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default= 500_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--gen_during_training", default=True, type=str2bool,
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=10, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--avg_model_beta", default=0.999, type=float, help="Average model beta.") #used for ema. # TODO check if it is ok. before was 0.9999
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument(
        "--grad_clip_norm",
        default=0.0,
        type=float,
        help="If > 0, clip gradient norm before optimizer step. Useful for one-hot/alignment stability.",
    )
    group.add_argument(
        "--skip_nonfinite_updates",
        default=True,
        type=str2bool,
        help="If True, skip backward/optimizer updates when loss or gradients are NaN/Inf.",
    )


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=6, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=7.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--recon_param", default=1e3, type=float,
                       help="Reconstruction parameter.")
    group.add_argument("--recon_step_start", default=-1, type=int, help="Highest step index to perform recon_guidance. Default -1 means from first step.")
    group.add_argument("--recon_step_stop", default=2, type=int, help="Lowest step index to perform recon_guidance. 0 means until the last step.")
    group.add_argument("--recon_frame_start", default=0, type=int, help="First frame index to perform recon_guidance.")
    group.add_argument("--recon_frame_stop", default=-1, type=int, help="Last frame index to perform recon_guidance. Default -1 means from last step.")
    group.add_argument("--sampling_mode", default='none', choices=['none', 'goal', 'traj', 'heading', 'heading_traj'], type=str, help="For classifier-guidance conditioning.")
    group.add_argument("--cfg_type", default='none', choices=['text', 'target', 'text_target'], type=str, help="For classifier-guidance conditioning.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")
    group.add_argument("--target_joint_source", default='data', choices=['data', 'random'], type=str, help="Either use targets from the data or choose random targets.")

def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    # group.add_argument("--prefix_end", default=0.25, type=float,
    #                    help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    # group.add_argument("--suffix_start", default=0.75, type=float,
    #                    help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--prefix_end", default=10, type=int,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=196, type=int,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")
    group.add_argument("--apply_cond", action='store_true',
                help="If True, will use EMA model averaging.")

def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", default='', type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--external_results_file", default='',type=str, 
                       help="Path to an npy file containing the external results of another model.")
    #group.add_argument("--do_unique", action='store_true', help="If true, select only one motion for each db key.")
    #group.add_argument("--eval_name", default='', type=str, help="Optional for wandb. if empty will use the model name instead.")
    #group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
    #                   help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
    #                        "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
    #                        "debug - short run, less accurate results."
    #                        "full (a2m only) - 20 repetitions.")
    #group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    #group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    #group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'],
    #                    help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    #group.add_argument("--guidance_param", default=7.5, type=float,
    #                   help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser) #OK
    #add_data_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args() #apply_rules(parser.parse_args())


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
