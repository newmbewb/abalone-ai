from dlabalone.data.data_processor import DataGenerator
from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.networks import simple1
from keras.models import Sequential
from keras.layers.core import Dense
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# out_filename_format = ''
# convert_dataset_to_npy(encoder=, batch=, in_dir=, out_filename_format=out_filename_format, test_ratio=, overwrite=False)
# generator = DataGenerator(out_filename_format) # If you want to change batch, you have to re-convert data
#
# ## when call fit_generator
# generator = generator.generator('train')
# steps_per_epoch=generator.get_num_steps('train')
# validation_data=generator.generator('test')
# validation_steps=generator.get_num_steps('test')

encoder = TwoPlaneEncoder(5)
generator = DataGenerator(encoder, 256, '../../dataset', '../../encoded_data', 0.5)

network_layers = simple1.layers(encoder.shape())
model = Sequential()
num_classes = encoder.num_moves()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
