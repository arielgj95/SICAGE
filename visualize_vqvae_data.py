import os
import pdb
import pickle

import numpy as np
import yaml
from pprint import pprint
from easydict import EasyDict
import torch
import math
import time
import torch.nn as nn


from vq_vae.vqvae import VQVAE
from TED4CL.process_poses import MotionPreprocessor
from dataset import load_samples,create_subject_dependent_split



global_vqvae_model = None
global_info_data = None
global_links_len = None
global_keypoint_indices = None
#device = torch.device('cuda:' + '0')


def _get_vqvae_core(model):
    return model.module if hasattr(model, "module") else model


def _has_module_prefix(state_dict):
    return any(str(k).startswith("module.") for k in state_dict.keys())


def _build_motion_preprocessor(skeleton_info, keypoint_indices):
    if isinstance(skeleton_info, dict):
        scene_fps = float(skeleton_info.get("fps", 30.0))
    else:
        scene_fps = 30.0
    return MotionPreprocessor(
        poses=[],
        scene=(0, 0),
        scene_fps=scene_fps,
        skeleton_info=skeleton_info,
        keypoint_indices=keypoint_indices,
    )


def _plot_motion(processor, path, save_name, n_frames):
    # Backward compatibility: some MotionPreprocessor versions expose only plot_poses_animation.
    if hasattr(processor, "plot_poses_animation_improved"):
        return processor.plot_poses_animation_improved(path=path, save_name=save_name, n_frames=n_frames)
    return processor.plot_poses_animation(path=path, save_name=save_name, n_frames=n_frames)


def evaluate_testset(model, test_data_loader, mydevice): #evaluation of the val set at the end of each epoch
    start = time.time()
    model = model.eval()
    euclidean_errors = []
    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            print(f"Processing {iter_idx} / {len(test_data_loader)-1}")
            pose_seq_eval = data[0].to(mydevice)  # （batch, 60, 54）
            labels = data[1]  # Not needed

            b, t, c = pose_seq_eval.size()
            output, loss, _ = model(pose_seq_eval)
            diff = (pose_seq_eval - output).view(b, t, c // 6, 6)
            euclidean_errors.append(torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=3))))
    print('generation took {:.2f} s'.format(time.time() - start))
    euclidean_errors = torch.stack(euclidean_errors)
    return euclidean_errors.mean().cpu().numpy(), euclidean_errors.std().cpu().numpy(), output.data.cpu(), pose_seq_eval.data.cpu(), labels

def generate_and_plot_pose(args, model_path,  test_loader, mydevice):

    #playlists_folder = "data_root"
    '''
    print("Loading data...")
    samples, culture_speakers, samples_counter = load_samples(data_path)
    print(culture_speakers)
    for culture in culture_speakers.keys():
        print(f"The culture {culture} has {len(culture_speakers[culture])}")
    train_loader, val_loader, test_loader = create_subject_dependent_split(samples, motion_only=True,
                                                                 batch_size=config.batch_size)

    print("Data loaded!")
    '''

    keypoint_indices = [7, 8, 9, 14, 15, 16, 11, 12, 13]

    links_len_file = getattr(args, "speaker_link_len", None)
    info_file = getattr(args, "skeleton_info_path", None)
    if not links_len_file or not info_file:
        raise ValueError(
            "VQ-VAE visualization requires --speaker_link_len and --skeleton_info_path."
        )

    with open(links_len_file,"rb") as file:
        links_len = pickle.load(file)

    with open(info_file,"rb") as file:
        info_data = pickle.load(file)
        info_data = info_data['meta_info']

    #train_loader, val_loader, test_loader = create_subject_dependent_split(samples, motion_only=True,
    #                                                             batch_size=config.batch_size)

    #Load the model
    with torch.no_grad():
        model = VQVAE(args.VQVAE, 9 * 6).to(mydevice)

        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_dict"])

        if torch.cuda.is_available() and hasattr(args, "gpus") and len(args.gpus) > 1:
            model = nn.DataParallel(model, device_ids=args.gpus, output_device=args.gpus[0])

        model = model.to(mydevice).eval()  # <-- make absolutely sure it’s on GPU

        print("Checkpoint loaded!")
        mean, std, output, input, in_labels = evaluate_testset(model, test_loader, mydevice)
        print('mean on test: {:.3f}, std on test: {:.3f}'.format(mean, std))

        b, t, c = output.shape
        input = input.view(b, t, c // 6, 6)
        #input = input * (std_data+1e6) + mean_data ##unnormalize data
        output = output.view(b, t, c // 6, 6)
        # poses, scene and scene_fps are just used for initialization. THey are not really necessary
        processor_in = MotionPreprocessor(poses=[], scene=(0,0), scene_fps= 30, skeleton_info=info_data, keypoint_indices = keypoint_indices)
        processor_in.create_parent_indices()
        processor_in.main_speaker_skeletons = input.cpu().numpy()
        processor_out = MotionPreprocessor(poses=[], scene=(0,0), scene_fps= 30, skeleton_info=info_data, keypoint_indices = keypoint_indices)
        processor_out.create_parent_indices()
        processor_out.main_speaker_skeletons = output.cpu().numpy()
        print("Starting the inference...")
        #output = output * (std_data+1e6) + mean_data ##unnormalize data
        for sample in range(10): #plot just 5 samples in the ba
            np.set_printoptions(threshold=np.inf, linewidth=200)

            in_poses = processor_in.rotation_6d_to_matrix(input[sample])
            #print("input 1:",input.shape,input)
            in_poses  = processor_in.convert_to_original_representation(in_poses)
            #print("SHAPE",input.shape,input)
            in_poses = processor_in.compute_joint_positions(in_poses, links_len)
            processor_in.main_speaker_skeletons = in_poses
            out_poses = processor_out.rotation_6d_to_matrix(output[sample])
            out_poses  = processor_out.convert_to_original_representation(out_poses)
            out_poses = processor_out.compute_joint_positions(out_poses,links_len)
            processor_out.main_speaker_skeletons = out_poses
            #print("INPUT TYPE",process_out.shape,type(process_out))

            print(f"plotting input motion of sample {sample}...")
            processor_in.plot_poses_animation( path=os.path.join(args.plot_path, "input_model_motion_plot_test"),
                             save_name=f"input_sample_{sample}",n_frames=len(in_poses))
            print(f"plotting output motion of sample {sample}...")
            processor_out.plot_poses_animation( path=os.path.join(args.plot_path, "output_model_motion_plot_test"),
                             save_name=f"output_sample_{sample}",n_frames=len(out_poses))

    #print("MEAN",mean,"STD",std, "output_shape:",output.shape,"input_shape:",input.shape)

def load_model(args, model_path):

    global global_vqvae_model, global_info_data, global_links_len, global_keypoint_indices
    keypoint_indices = getattr(args, "keypoint_indices", [7, 8, 9, 14, 15, 16, 11, 12, 13])

    # Reference skeleton paths must be supplied by the caller.
    links_len_file = getattr(args, "speaker_link_len", None)
    info_file = getattr(args, "skeleton_info_path", None)
    if not links_len_file or not info_file:
        raise ValueError(
            "Loading the VQ-VAE visualization model requires --speaker_link_len "
            "and --skeleton_info_path."
        )

    with open(links_len_file, "rb") as file:
        links_len = pickle.load(file)

    with open(info_file, "rb") as file:
        info_raw = pickle.load(file)
        info_data = info_raw.get("meta_info", info_raw) if isinstance(info_raw, dict) else info_raw

    gpu_arg = str(getattr(args, "gpu", "0"))
    if torch.cuda.is_available():
        mydevice = torch.device(f"cuda:{gpu_arg}")
    else:
        mydevice = torch.device("cpu")

    with torch.no_grad():
        model = VQVAE(args.VQVAE, 9 * 6)  # n_joints * n_channels
        if torch.cuda.is_available() and hasattr(args, "no_cuda"):
            device_ids = [eval(i) for i in args.no_cuda]
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(mydevice)

        checkpoint = torch.load(model_path, map_location=mydevice)
        state_dict = checkpoint.get("model_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported VQ-VAE checkpoint format at {model_path}")

        model_is_dataparallel = isinstance(model, nn.DataParallel)
        state_has_module_prefix = _has_module_prefix(state_dict)
        if model_is_dataparallel and not state_has_module_prefix:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif (not model_is_dataparallel) and state_has_module_prefix:
            state_dict = {
                (k[7:] if str(k).startswith("module.") else k): v
                for k, v in state_dict.items()
            }

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"[visualize_vqvae_data] Missing keys while loading checkpoint: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[visualize_vqvae_data] Unexpected keys while loading checkpoint: {incompatible.unexpected_keys}")
        model.eval()

    global_vqvae_model = model
    global_info_data = info_data
    global_links_len = links_len
    global_keypoint_indices = keypoint_indices

    return model,info_data,links_len,keypoint_indices

def sample_generation_from_codebooks(vqvae_args, generated_codebooks, real_codebooks,
                                     save_path_real, save_path_fake, step=None, plot_poses=True):
    # Load the model
    global global_vqvae_model, global_info_data, global_links_len, global_keypoint_indices
    if global_vqvae_model is None:
        model_path = vqvae_args.checkpoint_path
        load_model(vqvae_args, model_path)
    model = global_vqvae_model
    model_core = _get_vqvae_core(model)
    info_data = global_info_data
    links_len = global_links_len
    keypoint_indices = global_keypoint_indices

    b1, t1, c1 = generated_codebooks.shape
    b2, t2, c2 = real_codebooks.shape
    all_codebooks = torch.cat((generated_codebooks, real_codebooks), dim=0)
    model_device = next(model_core.parameters()).device
    all_codebooks = all_codebooks.to(model_device)
    codebook_vectors = all_codebooks.permute(0, 2, 1) # Convert to (batch_size, codebook_dim, t)
    all_generated_poses = []
    all_true_poses = [ ]

    # Decode using the bottleneck
    with torch.no_grad():
        zs = model_core.bottleneck.encode([codebook_vectors]) #encode to closest codebook indices
        #xs_quantised = model.module.bottleneck.decode(zs)
        decoded_motion =  model_core.decode(zs) # Use the existing `decode` function to get the final motion
        batch_size, n_poses, n_rotations = decoded_motion.shape
        #decoded_motion = decoded_motion.squeeze(0).data.cpu().numpy()
        decoded_motion = decoded_motion.view(batch_size, n_poses, n_rotations // 6, 6) # [batch_size, n_frames, n_joints, n_rotations_per_joint]
        decoded_fake = decoded_motion[:b1]
        decoded_real = decoded_motion[b1:]

        processor_fake = _build_motion_preprocessor(info_data, keypoint_indices)
        processor_fake.create_parent_indices()
        processor_fake.main_speaker_skeletons = decoded_fake.cpu().numpy()
        processor_real = _build_motion_preprocessor(info_data, keypoint_indices)
        processor_real.create_parent_indices()
        processor_real.main_speaker_skeletons = decoded_real.cpu().numpy()
        #np.set_printoptions(threshold=np.inf, linewidth=200)
        # output = output * (std_data+1e6) + mean_data ##unnormalize data
        for sample in range(b1):
            np.set_printoptions(threshold=np.inf, linewidth=200)
            # print("TEST",test_data['data'].shape)
            # print("SAMPLE",sample)
            decoded_fake_s = processor_fake.rotation_6d_to_matrix(decoded_fake[sample])
            # print("input 1:",input.shape,input)
            decoded_fake_s = processor_fake.convert_to_original_representation(decoded_fake_s.cpu().numpy())
            # print("SHAPE",input.shape,input)
            decoded_fake_s = processor_fake.compute_joint_positions(decoded_fake_s, links_len)
            processor_fake.main_speaker_skeletons = decoded_fake_s
            decoded_real_s = processor_real.rotation_6d_to_matrix(decoded_real[sample])
            decoded_real_s = processor_real.convert_to_original_representation(decoded_real_s.cpu().numpy())
            decoded_real_s = processor_real.compute_joint_positions(decoded_real_s, links_len)
            processor_real.main_speaker_skeletons = decoded_real_s
            # print("INPUT TYPE",process_out.shape,type(process_out))
            if plot_poses == True:
                print(f"plotting input motion of sample {sample}...")
                _plot_motion(
                    processor_real,
                    path=os.path.join(save_path_real, f"step_number_{step}"),
                    save_name=f"input_sample_{sample}",
                    n_frames=len(decoded_real_s),
                )
                print(f"plotting output motion of sample {sample}...")
                _plot_motion(
                    processor_fake,
                    path=os.path.join(save_path_fake, f"step_number_{step}"),
                    save_name=f"output_sample_{sample}",
                    n_frames=len(decoded_fake_s),
                )
            all_generated_poses.append(decoded_fake_s)
            all_true_poses.append(decoded_real_s)

    return all_true_poses, all_generated_poses



def main(args, source_pose, model_path, save_path, prefix, normalize=True):
    # source_pose = source_pose[:3600]        # 60s, 60FPS

    # normalize
    if normalize:
        data_mean = np.array(args.data_mean).squeeze()
        data_std = np.array(args.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)
        source_pose = (source_pose - data_mean) / std

    clip_length = source_pose.shape[0]

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses

    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / unit_time) + 1

    print('{}, {}, {}'.format(num_subdivision, unit_time, clip_length))

    with torch.no_grad():
        # model = VQVAE(args.VQVAE, 15 * 3)  # n_joints * n_chanels
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        # for i in range(0, num_subdivision):
        for i in range(0, 512):
            start_time = i * unit_time

            # prepare pose input
            pose_start = math.floor(start_time)
            pose_end = pose_start + args.n_poses
            in_pose = source_pose[pose_start:pose_end]
            if len(in_pose) < args.n_poses:
                if i == num_subdivision - 1:
                    end_padding_duration = args.n_poses - len(in_pose)
                in_pose = np.pad(in_pose, [(0, args.n_poses - len(in_pose)), (0, 0)], mode='constant')
            in_pose = torch.from_numpy(in_pose).unsqueeze(0).to(mydevice)
            # zs = model.module.encode(in_pose.float())
            # zs = [torch.arange(0, 512).unsqueeze(0).to(mydevice)]
            zs = [torch.tensor([i] * 30).unsqueeze(0).to(mydevice)]
            pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()
            # pose_sample, _, _ = model.module(in_pose.float())
            # pose_sample = pose_sample.squeeze(0).data.cpu().numpy()

            code.append(zs[0].squeeze(0).data.cpu().numpy())
            result.append(pose_sample)
            # break
    out_code = np.vstack(code)
    out_poses = np.vstack(result)

    if normalize:
        out_poses = np.multiply(out_poses, std) + data_mean
    print(out_poses.shape)
    print(out_code.shape)
    np.save(os.path.join(save_path, 'code' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'generate' + prefix + '.npy'), out_poses)
    return out_poses, out_code


def cal_distance(args, model_path, save_path, prefix, normalize=True):

    with torch.no_grad():
        # model = VQVAE(args.VQVAE, 15 * 3)  # n_joints * n_chanels
        #model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = VQVAE(args.VQVAE, 9 * 6)
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        # for i in range(0, num_subdivision):
        for i in range(0, 512):
            # prepare pose input
            zs = [torch.tensor([i] * 7).unsqueeze(0).to(mydevice)] #*30
            pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()
            code.append(zs[0].squeeze(0).data.cpu().numpy())
            result.append(pose_sample)
    # code: (512, 30)
    # poses: (512, 240, 135)
    np.savez_compressed('./output/code.npz', code=np.array(code), poses=np.array(result), signature=np.mean(np.array(result), axis=1))


def visualize_code(args, model_path, save_path, prefix, code_source, normalize=True):
    # source_pose = source_pose[:3600]        # 60s, 60FPS

    # normalize
    if normalize:
        data_mean = np.array(args.data_mean).squeeze()
        data_std = np.array(args.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)

    with torch.no_grad():
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        zs = [torch.from_numpy(code_source.flatten()).unsqueeze(0).to(mydevice)]
        pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()

        code.append(zs[0].squeeze(0).data.cpu().numpy())
        result.append(pose_sample)

    out_code = np.vstack(code)
    out_poses = np.vstack(result)

    if normalize:
        out_poses = np.multiply(out_poses, std) + data_mean
    print(out_poses.shape)
    print(out_code.shape)
    np.save(os.path.join(save_path, 'code' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'generate' + prefix + '.npy'), out_poses)
    return out_poses, out_code


def visualize_PCA_codebook(signature_path, pic_save_path):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    signature = np.load(signature_path)['signature']
    codebook_size = signature.shape[0]
    c2s = []
    print(codebook_size)
    for i in range(codebook_size):
        c2s.append(signature[i])

    pca = PCA()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])
    Xt = pipe.fit_transform(c2s)
    plt.figure()
    plt.scatter(Xt[:, 0], Xt[:, 1], label='code')
    plt.legend()
    plt.title("PCA of Codebook")
    plt.savefig(pic_save_path + 'PCA_w_scaler.jpg')


def visualize_code_freq(code, output_path):
    from collections import Counter
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 2))
    print(code.shape)
    code = code.flatten()

    result = Counter(code)
    result_sorted = sorted(result.items(), key=lambda item: item[1], reverse=True)

    x = []
    y = []
    for d in result_sorted[:50]:
        x.append(str(d[0]))
        y.append(d[1])

    p1 = plt.bar(x[0:len(x)], y[0:len(x)])
    plt.bar_label(p1, label_type='edge')
    plt.tight_layout()
    plt.savefig(output_path + '/visualize_code_freq_top15.jpg')


def clip_code_unit(video_path, save_path):
    import subprocess
    import os

    delta_X = 4  # 每10s切割

    mark = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 获取视频的时长
    def get_length(filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)

    min = int(get_length(video_path)) // 60  # file_name视频的分钟数
    second = int(get_length(video_path)) % 60  # file_name视频的秒数
    totol_sec = int(get_length(video_path))

    print(min, second, totol_sec)

    for i in range(0, totol_sec, delta_X):

        min_start = str(i // 60)
        start = str(i % 60)
        min_end = str((i + delta_X) // 60)
        end = str((i + delta_X) % 60)

        # crop video
        # 保证两位数
        if len(str(min_start)) == 1:
            min_start = '0' + str(min_start)
        if len(str(min_end)) == 1:
            min_end = '0' + str(min_end)
        if len(str(start)) == 1:
            start = '0' + str(start)
        if len(str(end)) == 1:
            end = '0' + str(end)

        # 设置保存视频的名字

        name = str(mark)
        command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -strict -2 {}'.format(
            video_path,
            min_start, start, min_end, end,
            os.path.join(save_path, name) + '.mp4')
        mark += 1
        os.system(command)


def pick_code_freq(train_code, code_int, topk=10, txt_dataset_path=None):
    from collections import Counter

    print(train_code.shape)

    line_code_count = {}
    for line in range(len(train_code)):
        result = Counter(train_code[line])
        line_code_count[line] = result[code_int]

    dataset = np.load(txt_dataset_path, allow_pickle=True)['aux']

    print([[dataset[i[0]],i[1]] for i in sorted(line_code_count.items(), key=lambda item: item[1], reverse=True)[:topk]])


def pick_code_txt(train_code, code_int=None, txt_dataset_path=None, stride=240, num_frames_code=30, fps=60, codebook_size=512, topk=3):
    from collections import Counter

    dataset = np.load(txt_dataset_path, allow_pickle=True)
    aux = dataset['aux']
    txt = dataset['txt']

    print(train_code.shape)

    code_txt = []

    # reshape txt
    step_sz = int(stride / num_frames_code)
    stride_time = stride//fps
    for line in txt:
        tmp_code_txt = [[] for _ in range(num_frames_code)]
        while line != []:
            tmp = line.pop(0)
            tmp_code_txt[int((tmp[0] % stride_time+ (tmp[1] % stride_time if tmp[1] % stride_time != 0 else stride_time)) * 60 / 2 / step_sz)].append(tmp)      # Prevent n*stride_time from being treated as 0
        code_txt.append(tmp_code_txt)

    # init code txt
    c2txt = {}
    txt2c = {}
    for i in range(codebook_size):
        c2txt[i] = []

    for i in range(train_code.shape[0]):       # for every stride
        for j in range(num_frames_code):        # for every code
            for tmp_code_txt in code_txt[i][(j-3 if j-3 > 0 else 0):(j+4 if j+4 < num_frames_code else num_frames_code)]:
                for tmp in tmp_code_txt:
                    c2txt[train_code[i][j]].append(tmp[2])
                    if tmp[2] not in txt2c:
                        txt2c[tmp[2]] = [train_code[i][j]]
                    else:
                        txt2c[tmp[2]].append(train_code[i][j])

    for i in range(codebook_size):
        count_txt = Counter(c2txt[i])
        count_txt = sorted(count_txt.items(), key=lambda item: item[1], reverse=True)[:topk]
        c2txt[i] = count_txt
        if count_txt == []:
            del c2txt[i]

    c2txt = sorted(c2txt.items(), key=lambda item: item[1][0][1], reverse=True)[:topk]

    # print(c2txt)

    for i in txt2c.keys():
        count_code = Counter(txt2c[i])
        count_code = sorted(count_code.items(), key=lambda item: item[1], reverse=True)[:topk]
        txt2c[i] = count_code

    pdb.set_trace()
    txt2c = sorted(txt2c.items(), key=lambda item: item[1][0][1], reverse=True)[:topk]




if __name__ == '__main__':
    from vq_vae.configs.parse_args import parse_args
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    mydevice = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    config.no_cuda = config.gpu
    #config.stage = "inference"
    #config.VQVAE_model_path = "vq_vae/output/train_codebook/codebook_checkpoint_best.bin"  # best instead of 100

    if config.stage == 'train':
        cal_distance(config, model_path=config.VQVAE_model_path, save_path=None,
                     prefix=None, normalize=True)
        code = np.load("vq_vae/output/code.npz")['code']
        output_path = "vq_vae/output"
        visualize_code_freq(code, output_path)

    elif config.stage == 'inference':
        lmdb_path = config.data_path
        if not lmdb_path:
            raise ValueError("Inference requires --data_path pointing to the motion-only LMDB.")
        metadata_pkl = os.path.join(lmdb_path, "metadata", "metadata.pkl")
        with open(metadata_pkl, "rb") as f:
            meta = pickle.load(f)

        sample_keys = meta["sample_keys"]
        speaker_encodings = meta["speaker_encodings"]
        culture_encodings = meta["culture_encodings"]

        splits_pkl = os.path.join(lmdb_path, "metadata", "splits_subject_dependent.pkl")

        train_loader, val_loader, test_loader = create_subject_dependent_split(
            sample_keys,
            motion_only=True,
            batch_size=config.batch_size,
            save_path=splits_pkl,
            lmdb_path=lmdb_path,
            speaker_encodings=speaker_encodings,
            culture_encodings=culture_encodings
        )
        print("Data Loaded!")
        print(config)
        generate_and_plot_pose(config, config.VQVAE_model_path, test_loader, mydevice)
        #visualizeCodeAndWrite(code_path=config.code_path, prefix=config.prefix, generateGT=False, save_path=config.save_path)
