import torch
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler
from torch.profiler import ProfilerActivity
from contextlib import contextmanager
from pytorch_lightning.utilities import rank_zero_only



class InferenceProfiler(SimpleProfiler):
    """
    This profiler records duration of actions with cuda.synchronize()
    Use this in test time.
    """

    def __init__(self):
        super().__init__()
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)

    @contextmanager
    def profile(self, action_name: str) -> None:
        try:
            torch.cuda.synchronize()
            self.start(action_name)
            yield action_name
        finally:
            torch.cuda.synchronize()
            self.stop(action_name)


def build_profiler(name, config):
    if name == 'inference':
        return InferenceProfiler()
    elif name == 'pytorch':
        from pytorch_lightning.profilers import PyTorchProfiler
        return PyTorchProfiler(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                               dirpath=f"./profiler_logs/{config['run']['experiment_name']}/{config['run']['run_name']}",
                               filename='pytorch_profile.txt',
                               profile_memory=True,
                               row_limit=100,
                               record_param_comms=False,
                               )
    elif name is None:
        return PassThroughProfiler()
    else:
        raise ValueError(f'Invalid profiler: {name}')