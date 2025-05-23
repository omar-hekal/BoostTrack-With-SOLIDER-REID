import torch

from solider_reid.config import cfg
from solider_reid.model import make_model


class SoliderReID(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        config_file = "external/solider_reid/configs/iust/swin_small.yml"
        cfg.merge_from_file(config_file)
        cfg.TEST.WEIGHT = weights_path
        self.model = make_model(cfg, num_class=0, camera_num=1, view_num=1, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        self.model.eval()
        self.model.cuda()

        self.model.load_param(cfg.TEST.WEIGHT)
        self.model = self.model.half()

    def forward(self, batch):
        with torch.no_grad():  # Ensure evaluation mode
            batch = batch.cuda().half()  # Move input batch to GPU
            output, _ = self.model(batch)  # Pass the batch through the model
        return output
        
