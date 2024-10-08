import torch.optim as optim

import sys
sys.path.append(r"../../")
from ...run_utils.callbacks.base import AccumulateRawOutput, PeriodicSaver, ProcessAccumulatedRawOutput,\
    ScalarMovingAverage, ScheduleLr, TrackLr, VisualizeOutput, TriggerEngine, ConditionalSaver
from ...run_utils.callbacks.logging import LoggingEpochOutput
from ...run_utils.engine import Events

from .targets import gen_targets, prep_sample
from .net_desc import create_model
from .run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output


def get_config(nr_type, mode, stroma_mask):
    return {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        "phase_list": [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "use_cpu": False,
                        "desc": lambda: create_model(
                            input_ch=3, nr_types=nr_type, 
                            freeze=True, mode=mode
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-5,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        #"lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "lr_scheduler": lambda x: optim.lr_scheduler.LambdaLR(x, lr_lambda=lambda epoch: min(1.0, epoch / 10)),
                        "extra_info": {
                            # loss function for each branch and associated weight
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        # "pretrained": -1,
                        "pretrained": "Useless crap",# "/mnt/gpid08/users/jose.perez.cano/checkpoints/4/01/net_epoch=50.tar",
                        # 'pretrained': None,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 4, "valid": 2,},  # engine name : value
                "nr_epochs": 50,
                "stroma_mask": stroma_mask,   # boolean indicating if stroma mask will be used
            },
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "use_cpu": False,
                        "desc": lambda: create_model(
                            input_ch=3, nr_types=nr_type, 
                            freeze=False, mode=mode
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-4,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        #"lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "lr_scheduler": lambda x: optim.lr_scheduler.LambdaLR(x, lr_lambda=lambda epoch: min(1.0, epoch / 10)),
                        "extra_info": {
                            # loss function for each branch and associated weight
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        "pretrained": -1,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 2, "valid": 2,}, # batch size per gpu
                "nr_epochs": 50,
                "stroma_mask": stroma_mask,  # boolean indicating if stroma mask will be used
            },
        ],
        # ------------------------------------------------------------------
        "run_engine": {
            "train": {
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": 1,  # number of threads for dataloader
                "run_step": train_step,  # TODO: function name or function variable ?
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(per_n_epoch=40),     # save checkpoints every "per_n_epoch"
                        # VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": 1,  # number of threads for dataloader
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        ProcessAccumulatedRawOutput(lambda a: proc_valid_step_output(a, nr_types=nr_type)),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }
