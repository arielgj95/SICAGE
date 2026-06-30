import torch
from mdm_generator.hierarchical_mdm import Hierarchical_MDM
from mdm_generator.diffusion import gaussian_diffusion as gd
from mdm_generator.diffusion.respace import SpacedDiffusion, space_timesteps
#from process_video.mdm_generator.diffusion.utils.parser_util import get_cond_mode


def load_model(model, state_dict, use_culture = True):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    # Positional encoders don't have anything to learn, so there is no need to load them as you can direclty apply positions without any paramters involved
    # TODO verify this. As I am using only one model, I will also load pos encoders as they don't make any difference
    state_dict.pop('gesture_pos_encoder.pe', None)  # no need to load it (fixed), and causes size mismatch for older models
    #del state_dict['embed_timestep.sequence_pos_encoder.pe'] #TODO this is needed only if I add positional encoding to timestep
    state_dict.pop('audio_pos_encoder.pe', None)
    if use_culture == False:
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'culture_embedder' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    else:
    #del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("UNEXPECTED KEYS: ", unexpected_keys)
    assert len(unexpected_keys) == 0
    #assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k for k in missing_keys])

# TODO remove as it is not called. Create gaussian diffusion is called in training loop
def create_model_and_diffusion(args, data):
    model = Hierarchical_MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

# TODO remove unnecesary paramters and put only the needed ones. It is also possible to avoid this part
# TODO ad in training_loop none of this stuff is used
def get_model_args(args, data):

    # default args
    #clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    #cond_mode = get_cond_mode(args) #the condition to the model, default is text #TODO I want it text + culture + audio
    if hasattr(data.dataset, 'num_actions'): #TODO remove this condition. My dataset doesn't have this
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    #TODO all this part is unnecesary. I already have the data; adapt the mdm model by passing the right parameters depending on my dataset (VQVAE codebooks)
    data_rep = 'rot6d'
    njoints = 25 # I jave 9 joint in my case. Notice that my input are VQVAE encodings and not poses
    nfeats = 6 #6 features to represent a joint rotation. Same in my dataset
    all_goal_joint_names = []

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
        all_goal_joint_names = ['pelvis'] + HML_EE_JOINT_NAMES
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    # Compatibility with old models
    if not hasattr(args, 'pred_len'): #pred len = 40. TODO change it to have an autoregressive model with pred_len length of predictions and context_len previous poses
        args.pred_len = 0
        args.context_len = 0

    emb_policy = args.__dict__.get('emb_policy', 'add') #concat is default. Now is add. Time and aux embeddings are either added or concatinated to the text embedings
    multi_target_cond = args.__dict__.get('multi_target_cond', False) #If true, enable multi-target conditioning
    multi_encoder_type = args.__dict__.get('multi_encoder_type', 'multi') #for multi joint condition. By default is single
    target_enc_layers = args.__dict__.get('target_enc_layers', 1)


    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'emb_before_mask': args.emb_before_mask, 'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'keyframe_cond_type': args.keyframe_cond_type,
            'pred_len': args.pred_len, 'context_len': args.context_len, 'emb_policy': emb_policy,
            'all_goal_joint_names': all_goal_joint_names, 'multi_target_cond': multi_target_cond, 'multi_encoder_type': multi_encoder_type, 'target_enc_layers': target_enc_layers,
            }



def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x_pred before applying noise,i.e. t=0, x_0_pred), that's our deal!
    steps = args.diffusion_steps #10
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta) #by default, cosine noise schedule
    loss_type = gd.LossType.MSE #Mean squared error between predicted and real poses #TODO add low_level ad high_level context

    if not timestep_respacing:
        timestep_respacing = [steps]

    #To train DiP without target conditioning, TODO analyze how the target affects the condition to understand how to use it
    # By default args.lambda_target_loc = 0 #
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.
    #it is exactly as GaussianDiffusion but can skip steps (in original case 1000 steps are used. Here just 10)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small #True by default
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma #False by default, so we keep gd.ModelVarType.FIXED_SMALL
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_unmasked_pose = args.lambda_unmasked_pose,
        lambda_masked_pose = args.lambda_masked_pose,
        lambda_culture = args.lambda_culture,
        lambda_low_level_context = args.lambda_low_level_context,
        lambda_high_level_context = args.lambda_high_level_context,
        lambda_diversity=args.lambda_diversity,
        lambda_contrastive=args.lambda_contrastive,
        use_gram_loss=getattr(args, "use_gram_loss", False),
        lambda_gram=getattr(args, "lambda_gram", 0.0),
        gram_eps=getattr(args, "gram_eps", 1e-6),
        lambda_collapse_reg=getattr(args, "lambda_collapse_reg", 0.0),
        use_alignment=getattr(args, "use_alignment", True),
        use_culture_loss=getattr(args, "use_culture_classification_layer", True),
        prefix_len = args.motion_prefix_len
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model(model, state_dict)
    return model
