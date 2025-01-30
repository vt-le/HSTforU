import torch
import torch.nn as nn
from models.enc.enc_615_1_pvt2 import pvt_v2_b2 as PVT2Encoder
from models.dec.dec_618_1_unet import UNetDec as UnetDecoder
from models.tem.tem_624_1_timesf_tempos import timeS_b1 as Temporal


class AnomalyDetection(nn.Module):
    def __init__(self, config, logger, is_trained=True):
        super().__init__()
        # Encoder
        self.encoder = PVT2Encoder()

        # TimeSformer
        self.temp = Temporal()

        # Decoder
        self.decoder = UnetDecoder()

        # Initialize
        if is_trained:
            self.encoder.init_weights(logger, config.MODEL.PRETRAINED)
            # self.decoder.init_weights(logger)

    def forward(self, x):
        features = []
        for xi in x:
            features.append(self.encoder(xi))

        # Transpose list of lists => https://stackoverflow.com/questions/6473679/transpose-list-of-lists
        features = list(map(list, zip(*features)))

        temporal = []
        for i in range(len(features)):
            temporal.append(torch.stack([fea for fea in features[i]], dim=1))

        temporal = self.temp(temporal)

        # TODO: There are many options here:
        # 1. repalce features by temporal
        # 2. features = features + temporal

        for i in range(len(features)):
            features[i] = torch.cat([fea for fea in features[i]], dim=1)

        # features + temporal
        for i in range(len(features)):
            features[i] = features[i] + temporal[i]

        # Decoder
        out = self.decoder(features)

        return out
