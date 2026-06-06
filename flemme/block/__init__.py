from flemme.block.common import *
from flemme.block.pcd import *
from flemme.logger import get_logger

logger = get_logger('flemme.block')


def get_building_block(name, **kwargs):
    logger.debug('building block parameters: {}'.format(kwargs))
    if name in ['single', 'conv']:
        return partial(ConvBlock, **kwargs)
    if name in ['double', 'double_conv']:
        return partial(DoubleConvBlock, **kwargs)
    if name in ['res', 'res_conv']:
        return partial(ResConvBlock, **kwargs)
    if name in ['dense', 'fc']:
        return partial(DenseBlock, **kwargs)
    if name in ['double_dense', 'double_fc']:
        return partial(DoubleDenseBlock, **kwargs)
    if name in ['res_dense', 'res_fc']:
        return partial(ResDenseBlock, **kwargs)
    if name in ['medmamba', 'lg_ssm_ffn']:
        return partial(MedMambaBlock, **kwargs)
    if name in ['lg_ssm', 'lg_ssm_non_ffn', 'medmamba_non_ffn']:
        return partial(MedMambaNonFFNBlock, **kwargs)
    logger.error(f'Unsupported building block: {name}')
    exit(1)
