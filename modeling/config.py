from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'Baseline'
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.IMAGE = CN()
_C.MODEL.BACKBONE.VIDEO = CN()
_C.MODEL.BACKBONE.TEXT = CN()
_C.MODEL.BACKBONE.IMAGE.NAME = 'resnet'
_C.MODEL.BACKBONE.IMAGE.STRIDE = 32
_C.MODEL.BACKBONE.IMAGE.PRETRAINED_PATH = './pretrained_models/resnet.pth'
_C.MODEL.BACKBONE.VIDEO.PRETRAINED_PATH = './pretrained_models/rgb_imagenet.pt'
_C.MODEL.BACKBONE.TEXT.FREEZE = False
_C.MODEL.BACKBONE.VIDEO.FREEZE = False
_C.MODEL.BACKBONE.SIDE_DATA_NAME = 'resnet18'
_C.MODEL.END_TO_END = False

_C.MODEL.BACKBONE.PRETRAIN = True
_C.MODEL.SYNC_BN = True
_C.MODEL.FUSION_TYPE = 'concat'
_C.MODEL.FREEZE_BN = True
_C.MODEL.USE_ABS_POS = False
_C.MODEL.NQ = 1

_C.MODEL.ATTENTION = CN()
_C.MODEL.ATTENTION.DIM = 256
_C.MODEL.ATTENTION.HEAD = 8
_C.MODEL.ATTENTION.USE_DECODER = False

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.TEMP = 0.1
_C.MODEL.LOSS.FOCAL = False
_C.MODEL.LOSS.BCE = True
_C.MODEL.LOSS.OVERALL_DICE = True
_C.MODEL.LOSS.MEAN_DICE = True
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.HEATMAP = False
_C.MODEL.HEAD.BOX_REGRESS = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = 'train'
_C.DATASETS.TEST = 'test'
_C.DATASETS.NAME = 'A2D'
_C.DATASETS.ROOT = './Dataset/A2D'

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.FRAME_PER_SIDE = 5
_C.INPUT.DYNAMIC_DOWNSAMPLE = False
_C.INPUT.DOWNSAMPLE = 3
_C.INPUT.USE_SIDE_DATA = False
_C.INPUT.END_TO_END = False  # input whole video
_C.INPUT.SEQUENCE_LENGTH = 50  # input whole video
_C.INPUT.WORD_FEAT = False
_C.INPUT.LONG_VIDEO = False
_C.INPUT.OVERLAP = 50
_C.INPUT.WORD = False
_C.INPUT.CLIP_LEN = 32
_C.INPUT.CLIP_LEN_COMPRESSED = 8
_C.INPUT.COMPRESSED = False

_C.INPUT.USE_INSTANCE = False
_C.INPUT.MAX_INSTANCE = 10
_C.INPUT.INSTANCE_MASK_NAME_TRAIN = ['instance_mask_rcnn_finetune2', 'instance_mask_rcnn']
_C.INPUT.INSTANCE_MASK_NAME_TEST = 'instance_mask_rcnn_finetune2'

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 5
_C.SOLVER.MILESTONES = [2, 4]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 5
_C.SOLVER.VAL_BATCH_SIZE = 5
_C.SOLVER.AMPE = True #automatic mixed precision training
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.CLIP_GRAD = 0.0
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.OPTIMIZER = 'SGD'

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.THRESHOLD = 0.5

_C.OUTPUT_DIR = 'output'
