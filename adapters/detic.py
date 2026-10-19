"""Construct the retained Detic implementation from explicit model resources."""


def create_segmentation(categories, *, gpu_id, config_file=None, checkpoint_file=None, category_frequency=None, clip_checkpoint=None):
    from lagmemo.perception.detection.detic.detic_perception import DeticPerception

    return DeticPerception(
        vocabulary="custom",
        custom_vocabulary="," + ",".join(categories),
        sem_gpu_id=gpu_id,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        category_frequency=category_frequency,
        clip_checkpoint=clip_checkpoint,
    )
