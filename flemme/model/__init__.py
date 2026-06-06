from flemme.model.sem import SegmentationModel as SeM
from flemme.model.ae import AutoEncoder as AE
from flemme.model.clm import ClassificationModel as ClM
from flemme.utils import load_config
from flemme.logger import get_logger
from flemme.encoder import create_encoder

logger = get_logger('model.create_model')

supported_models = {
    'ClM': ClM,
    'SeM': SeM,
    'AE': AE,
}


def create_model(model_config, create_encoder_fn=create_encoder, create_model_fn=None):
    tmpl_path = model_config.pop('template_path', None)
    if tmpl_path is not None:
        logger.info('creating model from template ...')
        model_config = load_config(tmpl_path).get('model')
        return create_model(model_config, create_encoder_fn=create_encoder_fn)

    model_name = model_config.pop('name', None)
    if model_name in supported_models:
        logger.info('creating MedMamba experiment model from specific configuration ...')
        return supported_models[model_name](model_config, create_encoder_fn=create_encoder_fn)

    logger.error(f'Unsupported model class: {model_name}, should be one of {supported_models.keys()}')
    exit(1)
