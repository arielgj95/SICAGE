from mdm_generator.diffusion.utils.parser_util import evaluation_parser
from mdm_generator.diffusion.utils.fixseed import fixseed
from datetime import datetime
from collections import OrderedDict
import  torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from mdm_generator.eval.metrics import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_activation_statistics,
    calculate_frechet_distance,
    calculate_diversity,
    calculate_multimodality,
)
#from closd.diffusion_planner.data_loaders.humanml.utils.utils import *
#from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model

from mdm_generator.diffusion import logger
from mdm_generator.diffusion.utils import dist_util
from mdm_generator.train_platforms import (
    ClearmlPlatform,
    TensorboardPlatform,
    NoPlatform,
    WandBPlatform,
)  # required for the eval operation

torch.multiprocessing.set_sharing_strategy('file_system')

# Evaluates how much do motion embeddings match some other embeddings, for isntance
# gestures, culture etc. TODO adapt it to compare motion with other embeddings
def evaluate_matching_score(motion_loaders, file):
    match_score_dict = defaultdict(lambda: {})
    R_precision_dict = defaultdict(lambda: {})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        matching_score_sum_text = 0
        matching_score_sum_audio = 0
        top_k_count_text = 0
        top_k_count_audio = 0
        all_size_text = 0
        all_size_audio = 0
        # print(motion_loader_name)
        with (torch.no_grad()):
            for idx, batch in enumerate(motion_loader):
                final_motion, culture_output, motion_pooled, low_level_context,\
                    high_level_context, overall_audio_context,labels = batch

                assert high_level_context.shape[0] == motion_pooled.shape[0], "Text and motion embeddings size mismatch"
                assert overall_audio_context.shape[0] == motion_pooled.shape[0], "Audio and motion embeddings size mismatch"

                # All data is related to 4 second of audio
                dist_mat_text = euclidean_distance_matrix(high_level_context.cpu().numpy(),
                                                          motion_pooled.cpu().numpy())
                dist_mat_audio = euclidean_distance_matrix(overall_audio_context.cpu().numpy(),
                                                           motion_pooled.cpu().numpy())
                matching_score_sum_text += dist_mat_text.trace()
                matching_score_sum_audio += dist_mat_audio.trace()

                argsmax_text = np.argsort(dist_mat_text, axis=1)
                argsmax_audio = np.argsort(dist_mat_audio, axis=1)
                top_k_mat_text = calculate_top_k(argsmax_text, top_k=3)
                top_k_mat_audio = calculate_top_k(argsmax_audio, top_k=3)
                top_k_count_text += top_k_mat_text.sum(axis=0)
                top_k_count_audio += top_k_mat_audio.sum(axis=0)

                all_size_text += high_level_context.shape[0]
                all_size_audio += overall_audio_context.shape[0]

                all_motion_embeddings.append(motion_pooled.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score_text = matching_score_sum_text / all_size_text
            matching_score_audio = matching_score_sum_audio / all_size_audio
            R_precision_text = top_k_count_text / all_size_text
            R_precision_audio = top_k_count_audio / all_size_audio
            match_score_dict[motion_loader_name]['audio'] = matching_score_audio
            match_score_dict[motion_loader_name]['text'] = matching_score_text
            R_precision_dict[motion_loader_name]['audio'] = R_precision_audio
            R_precision_dict[motion_loader_name]['text'] = R_precision_text
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score Audio: {matching_score_audio:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score Audio: {matching_score_audio:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] Matching Score Text: {matching_score_text:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score Text: {matching_score_text:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision Audio: '
        for i in range(len(R_precision_audio)):
            line += '(top %d) Audio: %.4f ' % (i+1, R_precision_audio[i])
        print(line)
        print(line, file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision Text: '
        for i in range(len(R_precision_text)):
            line += '(top %d) Text: %.4f ' % (i+1, R_precision_text[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict

# TODO use it after decoding the motion to understand if jerk compares
def compute_jerk(motion, frame_rate=30):
    """
    Computes the jerk of the motion.

    Args:
        motion (np.ndarray): Motion data of shape (num_frames, num_joints, num_dimensions).
        frame_rate (int): Frames per second.

    Returns:
        float: Average jerk over the motion.
    """
    dt = 1.0 / frame_rate
    velocity = np.gradient(motion, axis=0) / dt
    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt
    jerk_magnitude = np.linalg.norm(jerk, axis=(1, 2))  # Sum over joints and dimensions
    return np.mean(jerk_magnitude)


def compute_l2_distance(real_motion, generated_motion):
    """
    Computes the L2 distance between real and generated motions.

    Args:
        real_motion (np.ndarray): Real motion data of shape (num_frames, num_joints, num_dimensions).
        generated_motion (np.ndarray): Generated motion data of the same shape.

    Returns:
        float: Average L2 distance over all frames.
    """
    if real_motion.shape != generated_motion.shape:
        raise ValueError("Real and generated motions must have the same shape.")
    l2_distances = np.linalg.norm(real_motion - generated_motion, axis=(1, 2))  # Sum over joints and dimensions
    return np.mean(l2_distances)


# It evaluates FID between ground truth motion embeddings and generated motion embeddings
# It uses 2D embeddings as all the other functions
# TODO implement culture vs culture statistics
def evaluate_fid(gt_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(gt_loader):

            final_motion, culture_output, motion_pooled, low_level_context, \
                high_level_context, overall_audio_context, labels = batch

            gt_motion_embeddings.append(motion_pooled.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        if model_name == 'ground truth': #skip ground truth. Here we are comparing generated embeddings with ground truth, not gt with gt
            continue
        mu, cov = calculate_activation_statistics(motion_embeddings) #generated motion embeddings
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict

# It evaluates diversity in the following way:
# Given embeddings of dim (n_elements, dim_embedding), it constructs two vectors
# by random sampling diversity_times elements inside each element to get
# vec1 = (n_elements, diversity_times), vec2 = (n_elements, diversity_times)
# Then, it computes the norm between the two vectors
# It evaluates diversity for both real gestures and generated gestures
def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

# Similar to evaluate_diversity, but here we compare difference in generated motions
# from same input
# TODO I don't have this kind of data loader yet. I need to output generated motions
# many times from the same input
def evaluate_multimodality(mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    #mm_num_times = 0
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def evaluate_culture_classification(data_loaders, num_classes=None):
    """
    Evaluates culture classification performance by computing F1 Score, Balanced Accuracy,
    Accuracy, and ROC AUC.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        num_classes (int, optional): Number of classes in the classification task.
                                     If None, inferred from data.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    eval_dict = OrderedDict()
    for model_name, data_loader in data_loaders.items():
        if model_name == 'ground truth': #skip ground truth. Here we are comparing generated embeddings with ground truth, not gt with gt
            continue
        metrics = OrderedDict({
            'F1_Score': None,
            'Balanced_Accuracy': None,
            'Accuracy': None,
            'ROC_AUC': None
        })

        all_labels = []
        all_preds = []
        all_probs = []

        # Set the model to evaluation mode
        # Assuming this function is part of a class with a model attribute
        # If not, ensure that the model is in eval mode outside this function
        # model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                # Unpack the batch
                final_motion, culture_output, motion_pooled, low_level_context, \
                    high_level_context, overall_audio_context, labels = batch

                # Extract ground truth labels
                culture_real = labels.cpu().numpy()
                all_labels.extend(culture_real)

                # Process model outputs
                # Assuming culture_output is raw logits; adjust if it's already probabilities
                if culture_output.dim() == 1 or (culture_output.dim() == 2 and culture_output.size(1) == 1):
                    # Binary classification
                    probs = torch.sigmoid(culture_output).squeeze().cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                else:
                    # Multi-class classification
                    probs = torch.softmax(culture_output, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                    if num_classes is None:
                        num_classes = culture_output.size(1)

                all_preds.extend(preds)
                all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        #print("LABELS",all_labels)
        #print("PREDS",all_preds)

        # Determine if it's binary or multi-class classification
        is_binary = all_probs.ndim == 1 or all_probs.shape[1] == 1

        # Compute F1 Score
        if is_binary:
            f1 = f1_score(all_labels, all_preds, average='binary')
        else:
            f1 = f1_score(all_labels, all_preds, average='macro')

        # Compute Balanced Accuracy
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)

        # Compute Accuracy
        acc = accuracy_score(all_labels, all_preds)

        # Compute ROC AUC
        try:
            if is_binary:
                roc_auc = roc_auc_score(all_labels, all_probs)
            else:
                # Binarize the labels for multi-class ROC AUC
                if num_classes is None:
                    num_classes = len(np.unique(all_labels))
                all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
                # If the number of classes is 2, roc_auc_score expects a single column
                if all_labels_binarized.shape[1] == 1:
                    all_labels_binarized = np.hstack((1 - all_labels_binarized, all_labels_binarized))
                roc_auc = roc_auc_score(all_labels_binarized, all_probs, average='macro', multi_class='ovr')
        except ValueError as e:
            print(f"ROC AUC computation failed: {e}")
            roc_auc = None

        # Populate the evaluation dictionary
        metrics['F1_Score'] = f1
        metrics['Balanced_Accuracy'] = balanced_acc
        metrics['Accuracy'] = acc
        metrics['ROC_AUC'] = roc_auc

        # Assign the metrics to the model in eval_dict
        eval_dict[model_name] = metrics

    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(gt_loader, gen_loader, log_file, replication_times,
               diversity_times, mm_num_times, run_mm=False, eval_platform=None):

    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'CultureClassification': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            motion_loaders['generated'] = gen_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            # Evaluate Culture Classification
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            culture_score_dict = evaluate_culture_classification(motion_loaders) #for this task, gen loader and val loader have the same data, so only one is evaluated

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, subdict in mat_score_dict.items():
                if isinstance(subdict, dict): #Audio and Text
                    for subkey, item in subdict.items():
                        combined_key = f"{key}_{subkey}"
                        # Initialize nested dictionaries if not present
                        if combined_key not in all_metrics['Matching Score']:
                            all_metrics['Matching Score'][combined_key] = [item]
                        else:
                            all_metrics['Matching Score'][combined_key].append(item)

            for key, subdict in R_precision_dict.items():
                if isinstance(subdict, dict): #Audio and Text
                    for subkey, item in subdict.items():
                        combined_key = f"{key}_{subkey}"
                        # Initialize nested dictionaries if not present
                        if combined_key not in all_metrics['R_precision']:
                            all_metrics['R_precision'][combined_key] = [item]
                        else:
                            all_metrics['R_precision'][combined_key].append(item)

            for key, subdict in culture_score_dict.items(): #only gen_loader is evaluated as val_loader is the same
                if isinstance(subdict, dict): #All metrics
                    for subkey, item in subdict.items():
                        combined_key = f"{key}_{subkey}"
                        # Initialize nested dictionaries if not present
                        if combined_key not in all_metrics['CultureClassification']:
                            all_metrics['CultureClassification'][combined_key] = [item]
                        else:
                            all_metrics['CultureClassification'][combined_key].append(item)


            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                print(metric_name, model_name)
                # it makes sense only if it has been replicated more than one time
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                    
        # log results
        if eval_platform is not None:
            for k, v in mean_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        eval_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                            iteration=1, group_name='Eval')
                else:
                    eval_platform.report_scalar(name=k, value=v, iteration=1, group_name='Eval')
        
        return mean_dict

'''
if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    
    assert args.model_path != '' or args.external_results_file != '' and not (args.model_path != '' and args.external_results_file != ''), 'either model_path or external_results_file args should be specified!'
    args.external_mode = args.external_results_file != ''
 
    if args.external_mode:
        save_dir = os.path.dirname(args.external_results_file)
        log_file = os.path.join(save_dir, 'eval.log')
    else:
        niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
        log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
        if args.guidance_param != 1.:
            log_file += f'_gscale{args.guidance_param}'
        log_file += f'_{args.eval_mode}'
        log_file += '.log'
        save_dir = os.path.dirname(log_file)  # has not been tested with WandB

    print(f'Will save to log file [{log_file}]')

    eval_platform_type = eval(args.train_platform_type)
    eval_platform = eval_platform_type(save_dir, name=args.eval_name)
    eval_platform.report_args(args, name='Args')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 5
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 10 # about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()


    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    # added new features + support for prefix completion:
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval',
                                    hml_type=args.hml_type, fixed_len=args.context_len+args.pred_len, pred_len=args.pred_len, device=dist_util.dev(),
                                    autoregressive=args.autoregressive)

    num_actions = gen_loader.dataset.num_actions


    if args.external_mode:
        model = None
        diffusion = None
    else:
        logger.log("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, gen_loader)

        logger.log(f"Loading checkpoints from [{args.model_path}]...")
        load_saved_model(model, args.model_path, use_avg=args.use_ema)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdm_loader(args,
            model=model, diffusion=diffusion, batch_size=args.batch_size,
            ground_truth_loader=gen_loader, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, 
            max_motion_length=gt_loader.dataset.opt.max_motion_length, num_samples_limit=num_samples_limit, 
            scale=args.guidance_param, hml_type=args.hml_type
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, 
               diversity_times, mm_num_times, run_mm=run_mm, eval_platform=eval_platform)
    eval_platform.close()
'''
