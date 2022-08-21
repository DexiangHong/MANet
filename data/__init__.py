from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate

from .a2d_dataset import A2DDataset
from .a2d_dataset_compressed import A2DDatasetCompressed
from .refer_youtube_vos import ReferYouTubeVOSDataset
from .jhmdb_compressed import JHMDBSentencesDatasetCompressed
from .rvos_compressed import ReferYouTubeVOSDatasetCompressed


def build_dataloader(cfg, args, splits, is_train):
    assert len(splits) >= 1

    def build_dataset(split):
        if cfg.DATASETS.NAME.lower() == 'a2d':
            if cfg.INPUT.COMPRESSED:
                dataset = A2DDatasetCompressed(cfg, mode=split)
            else:
                dataset = A2DDataset(cfg, mode=split)
        elif cfg.DATASETS.NAME.lower() == 'youtube':
            dataset = ReferYouTubeVOSDatasetCompressed(cfg, mode=split)

        elif cfg.DATASETS.NAME.lower() == 'jhmdb':
            dataset = JHMDBSentencesDatasetCompressed(cfg)
        else:
            raise NotImplementedError
        return dataset

    dataset = build_dataset(splits)

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train, drop_last=True if is_train else False)
    else:
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)

    # collate_fn = (lambda x: x) if cfg.INPUT.END_TO_END else default_collate
    collate_fn = default_collate
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
    else:
        batch_size = cfg.SOLVER.VAL_BATCH_SIZE

    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=sampler,
                        drop_last=True,
                        collate_fn=collate_fn,
                        num_workers=cfg.SOLVER.NUM_WORKERS)

    return loader


