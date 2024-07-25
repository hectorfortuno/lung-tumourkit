import json
from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

# import sys
# sys.path.append(r"..")
from ..run_utils.utils import convert_pytorch_checkpoint

####
class InferManager(object):
    def __init__(self, **kwargs):

        """start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()""" 

        # Initialize variables
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.nr_types = self.method["model_args"]["nr_types"]
        self.stroma_mask = self.method["stroma_mask"]

        # Load the model (and the stroma one if needed)
        self.__load_model()
        if self.stroma_mask:
            self.__load_stroma_model()

        # Information of labels and corresponding name and colour
        self.type_info_dict = {None: ["no label", [0, 0, 0]],}      # default
        if self.nr_types is not None and self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.type_info_dict = {int(k): (v[0], tuple(v[1])) for k, v in self.type_info_dict.items()}
            # availability check
            for k in range(self.nr_types):
                if k not in self.type_info_dict:
                    assert False, "Not detect type_id=%d defined in json." % k

        if self.nr_types is not None and self.type_info_path is None:
            cmap = plt.get_cmap("hot")
            colour_list = np.arange(self.nr_types, dtype=np.int32)
            colour_list = (cmap(colour_list)[..., :3] * 255).astype(np.uint8)
            # should be compatible out of the box wrt qupath
            self.type_info_dict = {
                k: (str(k), tuple(v)) for k, v in enumerate(colour_list)
            }

        """end.record()
        torch.cuda.synchronize() 
        print(f'Model loading: {start.elapsed_time(end)/1000}')"""

        return

    def __load_model(self):
        """
        Create the model, load the checkpoint and define
        associated run steps to process each data batch.
        """

        # LOAD HOVERNET
        # model_desc = import_module(".segmentation.hovernet.models.hovernet.net_desc", package="tumourkit")
        model_desc = import_module(".hovernet.models.hovernet.net_desc", package="segmentation")
        model_creator = getattr(model_desc, "create_model")

        net = model_creator(**self.method["model_args"])
        if self.nr_gpus > 0:
            saved_state_dict = torch.load(self.method["model_path"])["desc"]
        else:
            saved_state_dict = torch.load(self.method["model_path"], map_location=torch.device('cpu'))["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        net.load_state_dict(saved_state_dict, strict=True)
        net = torch.nn.DataParallel(net)
        if self.nr_gpus > 0:
            net = net.to("cuda")

        # DEFINE FORWARD STEP AND POST-PROCESSING OF HOVERNET
        # module_lib = import_module(".segmentation.hovernet.models.hovernet.run_desc", package="tumourkit")
        module_lib = import_module(".hovernet.models.hovernet.run_desc", package="segmentation")
        run_step = getattr(module_lib, "infer_step")
        self.run_step = lambda input_batch: run_step(input_batch, net, use_cpu=self.nr_gpus==0)

        # module_lib = import_module(".segmentation.hovernet.models.hovernet.post_proc", package="tumourkit")
        module_lib = import_module(".hovernet.models.hovernet.post_proc", package="segmentation")
        self.post_proc_func = getattr(module_lib, "process")

        return

    def __load_stroma_model(self):

        # Stroma model
        path_networks = '/home/usuaris/imatge/sonia.rabanaque/KI67/VH22/Xarxes/Version2/'
        name_model = 'stroma_model_KI67.pth'
        stroma_model = torch.load(os.path.join(path_networks, name_model), map_location=torch.device('cpu'))
        stroma_model.segmentation_head._modules['2'] = nn.Identity()
        stroma_model.to("cuda")

        self.stroma_net = stroma_model

    def __save_json(self, path, old_dict, mag=None):
        new_dict = {}
        for inst_id, inst_info in old_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to jsonable
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict[int(inst_id)] = new_inst_info

        json_dict = {"mag": mag, "nuc": new_dict}  # to sync the format protocol
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return new_dict
