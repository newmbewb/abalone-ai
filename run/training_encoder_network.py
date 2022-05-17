import os
import time

from dlabalone.data.encoder_trainset_generator import get_encoder_trainset_filelist, encoder_trainset_generator, \
    generate_encoder_trainset
from dlabalone.encoders.alpha_abalone import AlphaAbaloneEncoder
from dlabalone.encoders.base import get_encoder_by_name
from dlabalone.encoders.fourplane import FourPlaneEncoder
from dlabalone.encoding_networks.simple1 import EncoderNetworkSimple1

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    batch = 4096
    steps = 64
    start_time = time.time()
    generate_encoder_trainset(FourPlaneEncoder(5, None), AlphaAbaloneEncoder(5, None), batch, steps,
                              '../data/encoder_trainset')
    print(time.time() - start_time)
    train_set, valid_set = get_encoder_trainset_filelist('../data/encoder_trainset')
    encoder_network = EncoderNetworkSimple1()
    input_encoder = get_encoder_by_name('fourplane', 5)
    output_encoder = get_encoder_by_name('alpha_abalone', 5, None)
    network_name = f'{encoder_network.name()}_{input_encoder.name()}_{output_encoder.name()}'
    model = load_model('../data/checkpoints/ACSimple1Policy_dropout0.1_AlphaAbaloneEncoder_epoch_1.h5')
    print(output_encoder.shape())
    model.summary()
    model = encoder_network.model(input_encoder.shape(), output_encoder.shape())
    model.summary()
    epochs = 1
    model.fit_generator(
        generator=encoder_trainset_generator(train_set),
        epochs=epochs,
        steps_per_epoch=len(train_set),
        validation_data=encoder_trainset_generator(valid_set),
        validation_steps=len(valid_set),
        callbacks=[
            ModelCheckpoint(f'../data/checkpoints/{network_name}_epoch_{{epoch}}.h5')
        ]
    )