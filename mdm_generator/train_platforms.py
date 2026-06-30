import os
import glob

def _create_summary_writer(save_dir):
    torch_tb_exc = None
    try:
        # Torch 1.10 uses `distutils.version` without importing submodule first.
        # Ensure attribute exists in Python 3.8 stdlib distutils package.
        import distutils
        if not hasattr(distutils, "version"):
            import distutils.version as _distutils_version
            distutils.version = _distutils_version

        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=save_dir)
    except Exception as torch_tb_exc:
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter(log_dir=save_dir)
        except Exception as tbx_exc:
            raise RuntimeError(
                "[TensorboardPlatform] No usable TensorBoard backend found. "
                f"torch.utils.tensorboard error: {torch_tb_exc}; "
                f"tensorboardX error: {tbx_exc}"
            ) from tbx_exc


class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='RL',
                              task_name=name)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        self.writer = _create_summary_writer(save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir, *args, **kwargs):
        pass

class WandBPlatform(TrainPlatform):
    import wandb
    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)
        self.wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
        self.wandb.init(
            project='RL',
            name=self.name,
            id=self.name,  # in order to send continued runs to the same record
            resume='allow',  # in order to send continued runs to the same record
            entity='tau-motion',
            save_code=True,
            config=config)  # config can also be sent via report_args()

    def report_scalar(self, name, value, iteration, group_name=None):
        self.wandb.log({name: value}, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        files = glob.glob(f'{local_path}/*.mp4')
        self.wandb.log({series: [self.wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        self.wandb.config.update(args)  #, allow_val_change=True)  # use allow_val_change ONLY if you want to change existing args (e.g., overwrite)

    def watch_model(self, *args, **kwargs):
        self.wandb.watch(args, kwargs)

    def close(self):
        self.wandb.finish()
