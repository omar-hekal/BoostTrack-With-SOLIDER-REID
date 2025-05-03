# Swin Small
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/iust/swin_small.yml MODEL.PRETRAIN_CHOICE 'self' MODEL.PRETRAIN_PATH 'swin_small.pth' SOLVER.BASE_LR 0.0002 SOLVER.OPTIMIZER_NAME 'SGD' MODEL.SEMANTIC_WEIGHT 0.2


