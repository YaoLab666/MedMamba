from . import pcd_transforms
from flemme.utils import DataForm, get_class, move_to_front, contains_one_of
from flemme.logger import get_logger

logger = get_logger('flemme.augment')


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

seg_label_transform_table = {
    'pcd': ['fixed', 'shuffle', 'totensor'],
}

random_transforms = ['FixedPoints', 'ShufflePoints']


def get_transforms(trans_config_list, data_form=DataForm.PCD, img_dim=None):
    transforms = []
    contain_fixed_points = False
    if len(trans_config_list) <= 0:
        return Compose(transforms)
    if data_form != DataForm.PCD:
        logger.error(f'Unsupported data form for MedMamba transforms: {data_form}')
        exit(1)

    contain_fixed_points = move_to_front(
        trans_config_list,
        func=lambda a: a['name'] == 'FixedPoints',
    )
    for tc in trans_config_list:
        trans_config = tc.copy()
        trans_name = trans_config.pop('name')
        trans_class = get_class(trans_name, pcd_transforms)
        if trans_class is None:
            logger.error(f"Unsupported point-cloud transform: {trans_name}")
            exit(1)
        transforms.append(trans_class(**trans_config))

    transforms = Compose(transforms)
    transforms.fixed_points = contain_fixed_points
    return transforms


def check_random_transforms(data_trans_config_list, label_trans_config_list):
    data_random_transform_list = [
        t['name'] for t in data_trans_config_list
        if t['name'] in random_transforms or 'random' in t['name'].lower() or 'noise' in t['name'].lower()
    ]
    label_random_transform_list = [
        t['name'] for t in label_trans_config_list
        if t['name'] in random_transforms or 'random' in t['name'].lower() or 'noise' in t['name'].lower()
    ]
    if len(data_random_transform_list) < len(label_random_transform_list):
        logger.warning('Label transforms introduce more random operations, are you sure this is what you want?')
    for drt, lrt in zip(data_random_transform_list, label_random_transform_list):
        if drt != lrt:
            logger.error('Transforms that introduce random operations should have the same order for data and label.')
            exit(1)


def check_consistent_transforms(data_trans_config_list, label_trans_config_list, data_form):
    if data_form != DataForm.PCD:
        logger.error(f'Unsupported data form for MedMamba transforms: {data_form}')
        exit(1)
    selector = seg_label_transform_table['pcd']
    data_consistent_transform_list = [
        t['name'] for t in data_trans_config_list
        if contains_one_of(t['name'].lower(), selector)
    ]
    label_consistent_transform_list = [
        t['name'] for t in label_trans_config_list
        if contains_one_of(t['name'].lower(), selector)
    ]
    if data_consistent_transform_list != label_consistent_transform_list:
        logger.error('Transforms that change the shape or order of data should be consistent.')
        exit(1)


def select_label_transforms(trans_config_list, data_form):
    if data_form != DataForm.PCD:
        logger.error(f'Unsupported data form for MedMamba transforms: {data_form}')
        exit(1)
    selector = seg_label_transform_table['pcd']
    return [t for t in trans_config_list if contains_one_of(t['name'].lower(), selector)]
