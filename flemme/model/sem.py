from flemme.model.base import BaseModel
from flemme.loss import get_loss

# segmentation model, don't has condition and time input
class SegmentationModel(BaseModel):
    def __init__(self, model_config, create_encoder_fn):

        super().__init__(model_config, create_encoder_fn)

        seg_loss_configs = model_config.get('segmentation_losses', [{'name':'Dice'}])
        if not type(seg_loss_configs) == list:
            seg_loss_configs = [seg_loss_configs, ]
            
        self.seg_losses = []
        self.seg_loss_names = []
        self.seg_loss_weights = []
        self.is_supervised = True
        for loss_config in seg_loss_configs:
            loss_config['reduction'] = self.loss_reduction
            self.seg_loss_weights.append(loss_config.pop('weight', 1.0))
            self.seg_loss_names.append(loss_config.get('name'))
            self.seg_losses.append(get_loss(loss_config, self.data_form))
        
        
    def __str__(self):
        _str = '********************* SeM ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def get_loss_name(self):
        return self.seg_loss_names
    def encode(self, x, c = None):
        return super().encode(x, c = c)
    def decode(self, z, c = None):
        return super().decode(z, c = c)
    def forward(self, x, c = None):
        seg_logits, z = super().forward(x, c = c, return_z = True)
        while type(z) == tuple or type(z) == list: 
            z = z[0]
        return {'seg_logits': seg_logits,
                'latent': z}
    def compute_loss(self, x, y, c = None):
        res = self.forward(x, c = c)
        losses = []
        for loss, weight in zip(self.seg_losses, self.seg_loss_weights):
            losses.append(loss(res['seg_logits'], y) * weight) 
        return losses, res
    
