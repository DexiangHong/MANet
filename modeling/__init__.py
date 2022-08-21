from .config import _C as cfg
from modeling.compressed_model_ms import CompressedModelMS
from modeling.compressed_model_ms_youtube import CompressedModelMSYoutube

def build_model(cfg):
    if cfg.MODEL.NAME.lower() == 'compressed_ms':
        model = CompressedModelMS(cfg)
    elif cfg.MODEL.NAME.lower() == 'compressed_youtube':
        model = CompressedModelMSYoutube(cfg)
    else:
        raise NotImplementedError

    return model