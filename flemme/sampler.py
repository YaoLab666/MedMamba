from flemme.logger import get_logger

logger = get_logger('sampler')


def create_sampler(model, sampler_config):
    logger.error('Sampler support was removed from the MedMamba point-cloud training package.')
    raise NotImplementedError('MedMamba classification, completion, and segmentation do not use generative sampling.')
