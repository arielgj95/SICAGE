import copy
import functools
import os
import time
import tqdm
from types import SimpleNamespace
import numpy as np

import re
from os.path import join as pjoin
from typing import Optional
import yaml
from torch.utils.data import Dataset, DataLoader, TensorDataset

import blobfile as bf
import torch
from easydict import EasyDict
from torch.optim import AdamW
from functools import partial
from torch.optim.lr_scheduler import StepLR

from mdm_generator.diffusion import logger
from mdm_generator.diffusion.utils import dist_util
from mdm_generator.diffusion.fp16_util import MixedPrecisionTrainer
from mdm_generator.diffusion.resample import LossAwareSampler, UniformSampler
from tqdm.auto import tqdm
from mdm_generator.diffusion.resample import create_named_schedule_sampler
from mdm_generator.eval import eval_ted4cl
from mdm_generator.diffusion.utils.model_util import load_model
from visualize_vqvae_data import sample_generation_from_codebooks

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
#INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, train_data, val_data):
        self.args = args
        self.train_platform = train_platform #Changed to TensorboardPlatform
        self.train_data = train_data
        self.val_data = val_data
        self.model = model
        self.model_avg = None
        if self.args.use_ema: #True by default
            self.model_avg = copy.deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.args.use_ema else self.model
        self.diffusion = diffusion
        logger.configure(dir=args.save_dir)

        with open(args.vqvae_config_path) as f:
            vq_vae_config = yaml.safe_load(f)
            self.vq_vae_config = EasyDict(vq_vae_config)

        self.pose_plot_generator = partial(sample_generation_from_codebooks)
        #self.cond_mode = model.cond_mode #There is no cond mode
        #self.data = data already put training and val sets
        self.batch_size = args.batch_size
        #self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval #1000 steps
        self.save_interval = args.save_interval #50000 steps
        self.resume_checkpoint = args.resume_checkpoint #"". if not empty, it loads checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay #0, no weight decay #TODO is it good?
        self.lr_anneal_steps = args.lr_anneal_steps #0

        self.step = 0
        self.resume_step = 0
        #self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps #600000 steps
        self.num_epochs = self.num_steps // len(self.train_data) + 1 #TODO train step accordingly to batch size dim

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        # Custom optimizer which can support MixedPrecision if needed. It supports the backward() func for
        # gradient descent using AdamW defined below. It updates the step and computes norm of paramters
        # for ema optimization
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            grad_clip_norm=getattr(args, "grad_clip_norm", 0.0),
            skip_nonfinite_updates=getattr(args, "skip_nonfinite_updates", True),
        )

        self.save_dir = args.save_dir  # e.g. data_root/hierachical_mdm_results
        self.overwrite = args.overwrite #True. If True, will enable to use an already existing save_dir

        if self.args.use_ema: #True
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters()
                 if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )
        self.scheduler = StepLR(self.opt, step_size=100000, gamma=0.5)  # Adjust parameters as needed #TODO try it
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev()) #sets the device

        self.schedule_sampler_type = 'uniform'
        # Cretes an uniform sampler
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        '''
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval',
                                            hml_type=self.args.hml_type, autoregressive=args.autoregressive,
                                            fixed_len=args.context_len+args.pred_len, pred_len=args.pred_len, device=dist_util.dev())

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_type=self.args.hml_type, 
                                                   hml_mode='gt', device=dist_util.dev())
            # TODO modify EvaluatiorMDMWrapper to suit my model. Possibly remove it as it just extracts embeddings
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader( self.args,
                    self.model_for_eval, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1., hml_type=args.hml_type,
                )
            }
        '''
        self.use_ddp = False
        self.ddp_model = self.model

    # ok
    def _load_and_sync_parameters(self):
        #find_resume_checkpoint finds the highest checkpoint path in args.save_dir
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint #if resume_checkpoint is not empty, this condition is true

        if resume_checkpoint:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            self.step += 1  
            # resumes the step number of the checkpoint
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint) 
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            #it is just a torch load to lad the parameters
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict: #so if I used ema
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model(self.model, state_dict)
                load_model(self.model_avg, state_dict_avg)
            else:
                load_model(self.model, state_dict)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)

            # self.model.load_state_dict(
            #     dist_util.load_state_dict(
            #         resume_checkpoint, map_location=dist_util.dev()
            #     ), strict=False
            # )

    #ok
    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    # TODO check why scaler is not defined
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True

    '''
    def cond_modifiers(self, cond, motion):
        # All modifiers must be in-place
        self.keyframes_modifier(cond, motion)
        self.spatial_cond_modifier(cond, motion)
        self.target_cond_modifier(cond, motion)

    
    def target_cond_modifier(self, cond, motion):
        if self.args.multi_target_cond:
            batch_size = motion.shape[0]
            cond['target_joint_names'], cond['is_heading'] = sample_goal(batch_size, motion.device, self.args.target_joint_names)

            cond['target_cond'] = get_target_location(motion, 
                                                      self.data.dataset.mean[None, :, None, None], 
                                                      self.data.dataset.std[None, :, None, None], 
                                                      cond['lengths'], 
                                                      self.data.dataset.t2m_dataset.opt.joints_num, self.model.all_goal_joint_names, cond['target_joint_names'], cond['is_heading']).detach()


    def spatial_cond_modifier(self, cond, motion):
        if self.args.spatial_condition is not None:
            if self.args.spatial_condition == 'traj':
                cond['condition_mask'] = torch.tensor(humanml_utils.HML_ROOT_HORIZONTAL_MASK[None, :, None, None])
            else:
                raise ValueError(f'unsupported spatial_condition [{self.args.spatial_condition}]')                 

    def keyframes_modifier(self, cond, motion):
        if self.args.keyframe_cond_type == '':
            return
        elif self.args.keyframe_cond_type == 'prefix':
            cond['clean_motion'] = copy.deepcopy(motion)
            is_prefix_cond = torch.bernoulli(torch.ones_like(cond['lengths']) * self.args.keyframe_cond_prob)  # 1-> keyfrane mode, 0-> regular mode
            cond['keyframe_lengths'] = (torch.rand_like(cond['lengths'].float()) * cond['lengths'] * is_prefix_cond).long()
            cond['keyframe_mask'] = lengths_to_mask(cond['keyframe_lengths'], cond['mask'].shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
        else:
            raise ValueError(f'unsupported keyframe_cond_type [{self.args.keyframe_cond_type}]')
    '''
    
    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for batch in tqdm(self.train_data, position=0, leave=True):
                motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels = batch
                #total step is current step + resume step. as lr_anneal_steps is 0 and self.total_step() < self.lr_anneal_steps is False
                #then this condition is always skipped
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break
                cultures = labels['culture_enc']
                #speakers = labels['speaker_enc']
                inputs_in_device = []
                for input in batch[:-1]:
                    inputs_in_device.append(input.to(self.device))
                #self.cond_modifiers(cond['y'], motion) # Modify in-place for efficiency
                #motion = motion.to(self.device)
                #cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                # run training step, updates the loss, the optimizer etc.
                self.run_step(inputs_in_device,cultures)
                # log the result
                if self.total_step() % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')
                if self.total_step() % self.save_interval == 0:
                    self.save() #save the model in self.save_dir + checkpoint name
                    if not self.args.eval_during_training: #true by default
                        pass
                    else:
                        self.model.eval() #put the model in evaluation mode
                        self.generate_during_training() #generates validation samples
                        self.evaluate() #evaluates the model and plots some samples
                        # Empty the data used for evaluation for unnecessary memory usage
                        self.generated_samples = []
                        self.real_samples = []
                        self.val_data_loader_generated = None
                        self.val_data_loader_real = None
                        self.plot_data = None
                    self.model.train() #put again the model in training mode

                    # Run for a finite amount of time in integration tests.
                    #if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                    #    return
                self.step += 1
            #lr_anneal_steps is 0 so I don't break
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.total_step() - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training: #true by default
            return
        start_eval = time.time()
        print('Running evaluation loop...')
        log_file = os.path.join(self.save_dir, f'eval_model{(self.total_step()):09d}.log')
        diversity_times = 300
        # This means that evaluate_multimodality is not evaluated during training
        mm_num_times = 0  # mm is super slow hence we won't run it during training
        eval_rep_times = 5
        self.plot_generated_samples()
        eval_dict = eval_ted4cl.evaluation(self.val_data_loader_real, self.val_data_loader_generated, log_file,
                                           replication_times=eval_rep_times ,
                                           diversity_times=diversity_times,
                                           mm_num_times=mm_num_times,
                                           run_mm=False, eval_platform=self.train_platform)
        print(eval_dict)
        for k, v in eval_dict.items():
            if k.startswith('R_precision'):
                for i in range(len(v)):
                    self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                      iteration=self.total_step(),
                                                      group_name='Eval')
            else:
                self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(),
                                                  group_name='Eval')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, inputs, cultures):
        has_finite_loss = self.forward_backward(inputs,cultures)
        took_step = False
        if has_finite_loss:
            took_step = self.mp_trainer.optimize(self.opt) #updates the state of the optimizer, compute norm of parameters, log the state, return True
        else:
            self.mp_trainer.zero_grad()
        if took_step:
            self.update_average_model() #just if ema is used
        self._anneal_lr() #lr_anneal_steps is 0 so it does nothing
        self.log_step() #logs the result

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, inputs,cultures):
        #print("FORWARD")
        #print(f"Memory before backward: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        self.mp_trainer.zero_grad() #set the gradients of all model parameters to zero
        #print(f"Memory after backward: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        motion = inputs[0]        # [batch_size, 25, 512]
        #text_features = inputs[1] # [batch_size, 768]
        #audio_mels = inputs[2]    # [batch_size, 156, 64]
        #audio_onsets = inputs[3]  # [batch_size, 156]
        #audio_wav2vec = inputs[4] # [batch_size, 50, 1024]
        #for i in range(0, batch.shape[0], self.microbatch):
        # Eliminates the microbatch feature
        #    assert i == 0
        #    assert self.microbatch == self.batch_size
        #    micro = batch
        #    micro_cond = cond
        #    last_batch = (i + self.microbatch) >= batch.shape[0]

        # it already knows how many timesteps are. Just pass the batch size
        t, weights = self.schedule_sampler.sample(motion.shape[0], dist_util.dev())

        # TODO pay attention to micro_cond. It the default behaviour, it contains whether mask
        # TODO should be used or not. I can keep mask and possibly add some other terms for the loss
        # functools.partial instantiates self.diffusion.training_losses with the following arguments
        # TODO add conditions as they can be needed to compute losses
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            inputs, #inputs are already on the device
            cultures.to(self.device),
            t,  # [bs](int) sampled timesteps
        )

        # forward and compute losses
        '''
        if last_batch or not self.use_ddp:
            losses = compute_losses()
        else:
            # TODO remove no_sync() as it is used in distributed computing to avoid gradient syncronization
            # TODO during micro_batch computation (i.e. a batch is divided in micro_batches, with no_sync
            # TODO we do not update gradient until all micro-batches are computed so we can for a big batch)
            with self.ddp_model.no_sync():
                losses = compute_losses()
        '''
        losses = compute_losses()

        # It is not. It is Uniform
        '''
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        '''

        loss = (losses["final_loss"] * weights).mean() #weights are all one.
        #print("LOOOOOOOOSSSS",loss)
        if not torch.isfinite(loss).item():
            bad_keys = [
                k for k, v in losses.items()
                if torch.is_tensor(v) and not torch.isfinite(v).all().item()
            ]
            logger.log(
                f"Skipping step {self.total_step()} because final_loss is non-finite. "
                f"Non-finite components: {bad_keys}"
            )
            logger.logkv_mean("skipped_nonfinite_loss", 1.0)
            return False
        logger.logkv_mean("skipped_nonfinite_loss", 0.0)
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        # mp_trainer is a custom optimizer which can be either or not in mixed precision (it is not by default)
        # when calling backward(loss) just calls loss.backwards() if fp32 is used
        self.mp_trainer.backward(loss)
        return True

    # lr_anneal_steps default is none
    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.batch_size)


    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

    def generate_during_training(self):
        self.generated_samples = []
        self.real_samples = []
        self.val_data_loader_generated = None
        self.val_data_loader_real = None
        self.plot_data = None

        # Set the model to evaluation mode
        self.ddp_model.eval()

        # Disable gradient calculations
        with torch.no_grad():
            #for idx, batch in enumerate(tqdm(self.val_data)):
            idx = 0
            for batch in tqdm(self.val_data, desc="Creating samples from validation data...", position=1, leave=True): #, desc="Sample generation from validation data"
                # Unpack the batch
                if idx == 500: #beak to 520 with batch_size = 64 as it starts to become too slow after
                    break
                motion, text_features, audio_mels, audio_onsets, audio_wav2vec, labels = batch
                culture_real = labels['culture_enc'].to(self.device)  # Assuming 'culture' is part of labels

                # Split motion into prefix and start
                x_prefix = motion[:, :self.args.motion_prefix_len, :].to(self.device)
                x_start = motion[:, self.args.motion_prefix_len:, :].to(self.device)
                in_data = [batch[i].to(self.device) for i in range(len(batch)-1)]
                in_data.append(culture_real)
                motion = in_data[0]

                new_motion, all_outputs = self.diffusion.p_sample_loop(self.ddp_model,in_data, clip_denoised=False) # generate samples with denoising loop
                '''
                # Generate noise
                noise = torch.randn_like(x_start)
                
                
                # Sample timesteps and weights
                t, weights = self.schedule_sampler.sample(x_start.shape[0], dist_util.dev())

                # Apply diffusion to get x_t
                x_t = self.diffusion.q_sample(x_start, t, noise=noise)
                

                # Concatenate prefix with x_t
                new_motion = torch.cat((x_prefix, x_t), dim=1)
                

                # Prepare data tuple
                data = (new_motion, text_features, audio_mels, audio_onsets, audio_wav2vec)

                # Forward pass through the model. Note that no masks are applied if self.ddp_model.eval(), so
                # motion_mask and audio_mask are all False. TODO verify this
                (motion_output, culture_output, motion_output_projection_pooled,
                 low_level_context, high_level_context, motion_mask, audio_mask,
                 overall_audio_context) = self.ddp_model(data, self.diffusion._scale_timesteps(t))
                '''
                motion_output = new_motion[:, self.args.motion_prefix_len:, :]
                culture_output = all_outputs[1]
                motion_output_projection_pooled = all_outputs[2]
                low_level_context = all_outputs[3]
                high_level_context = all_outputs[4]
                overall_audio_context = all_outputs[-1]

                # Prepare generated batch
                batch_generated = (
                    motion_output.detach().clone(),
                    culture_output.detach().clone(),
                    motion_output_projection_pooled.detach().clone(),
                    low_level_context.detach().clone(),
                    high_level_context.detach().clone(),
                    overall_audio_context.detach().clone(),
                    culture_real.detach().clone()  # All data are related to 4 seconds of motion
                )

                # Pool real motion data using the pooler of the model (we assume it is good at reducing
                # temporal dimension even on real data. Note that this is not completely realistic so we may have
                # some slight inconsistency during high level features alignment evaluation)
                new_motion_pooled = self.ddp_model.output_projection_pooler(x_start)

                # Prepare real batch
                real_batch = (
                    x_start.detach().clone(),
                    culture_output.detach().clone(),  # Use the actual labels here if needed
                    new_motion_pooled.detach().clone(),
                    low_level_context.detach().clone(),
                    high_level_context.detach().clone(),
                    overall_audio_context.detach().clone(),
                    culture_real.detach().clone()
                )
                if idx == 0: #one batch is sufficient. We just plot few samples
                    #for plotting generated and real motion
                    plot_batch = (new_motion, motion)
                    self.plot_data = plot_batch

                # Append to respective lists
                self.generated_samples.append(batch_generated)
                self.real_samples.append(real_batch)
                idx += 1

        # After generating all samples, create data loaders. Fills self.val_data_loader_generated and self.val_data_loader_real
        self._create_val_data_loaders()


    def _create_val_data_loaders(self):
        # Helper function to stack tensors and create DataLoaders

        # ----------------------------
        # Process Generated Samples
        # ----------------------------
        if self.generated_samples:
            # Unpack and concatenate each field across all batches
            gen_motion_output = torch.cat([batch[0] for batch in self.generated_samples], dim=0)
            gen_culture_output = torch.cat([batch[1] for batch in self.generated_samples], dim=0)
            gen_motion_output_projection_pooled = torch.cat([batch[2] for batch in self.generated_samples], dim=0)
            gen_low_level_context = torch.cat([batch[3] for batch in self.generated_samples], dim=0)
            gen_high_level_context = torch.cat([batch[4] for batch in self.generated_samples], dim=0)
            gen_overall_audio_context = torch.cat([batch[5] for batch in self.generated_samples], dim=0)
            gen_labels = torch.cat([batch[6] for batch in self.generated_samples], dim=0)
            # Handle labels if needed (optional)

            # Create a TensorDataset
            generated_dataset = GeneratedDataset(
                gen_motion_output,
                gen_culture_output,
                gen_motion_output_projection_pooled,
                gen_low_level_context,
                gen_high_level_context,
                gen_overall_audio_context,
                gen_labels
            )


            # Create a DataLoader
            self.val_data_loader_generated = DataLoader(
                generated_dataset,
                batch_size=32,
                shuffle=False,
            )

        # ----------------------------
        # Process Real Samples
        # ----------------------------
        if self.real_samples:
            # Unpack and concatenate each field across all batches
            real_x_start = torch.cat([batch[0] for batch in self.real_samples], dim=0)
            real_culture_real = torch.cat([batch[1] for batch in self.real_samples], dim=0)
            real_motion_pooled = torch.cat([batch[2] for batch in self.real_samples], dim=0)
            real_low_level_context = torch.cat([batch[3] for batch in self.real_samples], dim=0)
            real_high_level_context = torch.cat([batch[4] for batch in self.real_samples], dim=0)
            real_overall_audio_context = torch.cat([batch[5] for batch in self.real_samples], dim=0)
            real_labels = torch.cat([batch[6] for batch in self.real_samples], dim=0)
            # Handle labels if needed (optional)

            # Create a TensorDataset
            real_dataset = GeneratedDataset(
                real_x_start,
                real_culture_real,
                real_motion_pooled,
                real_low_level_context,
                real_high_level_context,
                real_overall_audio_context,
                real_labels
            )

            # Create a DataLoader
            self.val_data_loader_real = DataLoader(
                real_dataset,
                batch_size=32,
                shuffle=False,
            )
            logger.log("Validation Datasets were generated!!")

    def plot_generated_samples(self):
        """
        Generates and plots samples during training by comparing real and generated motions.
        """
        # Set the model to evaluation mode
        self.ddp_model.eval()

        # Ensure the directory for saving plots exists
        save_dir_plots = os.path.join(self.save_dir, "plots_during_training")
        os.makedirs(save_dir_plots, exist_ok=True)

        # Select the first batch from validation data
        generated_motion = self.plot_data[0][:self.args.gen_num_samples]
        real_motion = self.plot_data[1][:self.args.gen_num_samples]

        # Define save paths
        step = self.total_step()
        save_path = os.path.join(save_dir_plots, f"step_{step}")
        save_path_real = os.path.join(save_path, "real_samples")
        save_path_generated = os.path.join(save_path, "generated_samples")

        # Create directories for real and generated samples
        os.makedirs(save_path_real, exist_ok=True)
        os.makedirs(save_path_generated, exist_ok=True)

        # Generate and save plots
        self.pose_plot_generator(self.vq_vae_config, generated_motion, real_motion, save_path_real, save_path_generated, step = step)

        # Optionally, set the model back to training mode if necessary
        # self.ddp_model.train()



    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (already existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2: #It should be 2 as if there is the word "model" inside
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


class GeneratedDataset(Dataset):
    def __init__(self, *samples):
        """
        Initialize the dataset with the list of generated samples.
        """
        self.samples = samples

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.samples[0].size(0)

    def __getitem__(self, index):
        """
        Retrieve a sample by index.
        """
        return tuple(tensor[index] for tensor in self.samples)

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    #ts are timesteps. There are 10 timesteps. This function given all the losses in the batch, and some timesteps
    #shows the mean loss and the loss for each t quartile where t belongs. With this, we can have insights if some timesteps
    # perform better or not, so we can adjust the model accordingly.
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
