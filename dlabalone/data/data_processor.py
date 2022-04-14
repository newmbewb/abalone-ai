import os
import multiprocessing
import queue
import struct
import time

from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.utils import load_file_board_move_pair
from tensorflow.keras.utils import to_categorical
import numpy as np


def encode_data_worker(encoder, q_infile, q_out):
    num_classes = encoder.num_moves()
    while True:
        try:
            file = q_infile.get_nowait()
        except queue.Empty:
            break
        pair_list = load_file_board_move_pair(file)
        print(file)
        for board, move in pair_list:
            label = encoder.encode_move(move)
            label = to_categorical(label, num_classes)
            feature = encoder.encode_board(board)

            feature = np.expand_dims(feature, axis=0)
            label = np.expand_dims(label, axis=0)

            q_out.put((feature, label))


def _convert_dataset_to_npy(encoder, batch, file_list, out_filename_feature, out_filename_label):
    # Put file names & Start threads to encode data
    q_infile = multiprocessing.Queue()
    q_data = multiprocessing.Queue()

    for file in file_list:
        q_infile.put_nowait(file)

    thread_count = os.cpu_count() - 1
    thread_list = []
    for _ in range(thread_count):
        p = multiprocessing.Process(target=encode_data_worker, args=(encoder, q_infile, q_data))
        p.start()
        thread_list.append(p)

    # Get encoded data and do work
    all_thread_end = False
    buffer_feature = []
    buffer_label = []
    fd_feature = open(out_filename_feature, 'wb')
    fd_label = open(out_filename_label, 'wb')
    fd_feature.write(struct.pack('<L', 0))
    fd_label.write(struct.pack('<L', 0))
    epochs_count = 0
    while not all_thread_end:
        # If all encoder worker finished, exit this loop after processing remain things
        all_thread_end = True
        for p in thread_list:
            if p.is_alive():
                all_thread_end = False
                break

        while not q_data.empty():
            # Do work:
            feature, label = q_data.get()
            buffer_feature.append(feature)
            buffer_label.append(label)
            if len(buffer_feature) == batch:
                features = np.concatenate(buffer_feature, axis=0)
                labels = np.concatenate(buffer_label, axis=0)
                buffer_feature = []
                buffer_label = []
                np.save(fd_feature, features)
                np.save(fd_label, labels)
                epochs_count += 1

        time.sleep(0.01)
    # Save ggodari
    if len(buffer_feature) > 0:
        features = np.concatenate(buffer_feature, axis=0)
        labels = np.concatenate(buffer_label, axis=0)
        np.save(fd_feature, features)
        np.save(fd_label, labels)
        epochs_count += 1

    # Save epochs count
    fd_feature.seek(0)
    fd_label.seek(0)
    fd_feature.write(struct.pack('<L', epochs_count))
    fd_label.write(struct.pack('<L', epochs_count))
    fd_feature.close()
    fd_label.close()


def convert_dataset_to_npy(encoder, batch, in_dir, out_filename_format, test_ratio, overwrite=False):
    # Count file
    data_files = [f for f in map(lambda x: os.path.join(in_dir, x), os.listdir(in_dir)) if os.path.isfile(f)]
    file_count = len(data_files)
    train_file_count = int(file_count * (1 - test_ratio))

    # Separate file for train and test
    train_files = data_files[:train_file_count]
    test_files = data_files[train_file_count:]

    out_filename_feature = out_filename_format % ('train', 'feature')
    out_filename_label = out_filename_format % ('train', 'label')
    if overwrite or (not os.path.isfile(out_filename_feature) or not os.path.isfile(out_filename_label)):
        print('Converting train data to npy...')
        _convert_dataset_to_npy(encoder, batch, train_files, out_filename_feature, out_filename_label)
        print('Converting train data to npy done.')
    else:
        print('Train npy already exists. Skipping converting')

    out_filename_feature = out_filename_format % ('test', 'feature')
    out_filename_label = out_filename_format % ('test', 'label')
    if overwrite or (not os.path.isfile(out_filename_feature) or not os.path.isfile(out_filename_label)):
        print('Converting test data to npy...')
        _convert_dataset_to_npy(encoder, batch, test_files, out_filename_feature, out_filename_label)
        print('Converting test data to npy done.')
    else:
        print('Test npy already exists. Skipping converting')


class DataGenerator:
    def __init__(self, encoder, batch, dataset_dir, out_dir, test_ratio, out_filename_format=None, overwrite=False):
        if out_filename_format is None:
            out_filename_format = f'{encoder.name()}_{batch}batch_{test_ratio}test_%s_%s.npy'
        out_filename_format = os.path.join(out_dir, out_filename_format)
        convert_dataset_to_npy(encoder, batch, dataset_dir, out_filename_format, test_ratio, overwrite)
        self.out_filename_format = out_filename_format
        self.epochs_count = {}
        self.fd_feature = {}
        self.fd_label = {}
        self._reset_file('train')
        self._reset_file('test')

    def _reset_file(self, work):
        if work in self.fd_feature:
            self.fd_feature[work].close()
        if work in self.fd_label:
            self.fd_label[work].close()
        self.fd_feature[work] = open(self.out_filename_format % (work, 'feature'), 'rb')
        self.fd_label[work] = open(self.out_filename_format % (work, 'label'), 'rb')

        epochs_count_feature = struct.unpack('<L', self.fd_feature[work].read(struct.calcsize('<L')))[0]
        epochs_count_label = struct.unpack('<L', self.fd_label[work].read(struct.calcsize('<L')))[0]
        assert epochs_count_feature == epochs_count_label, 'Feature and label size are different'
        self.epochs_count[work] = epochs_count_feature

    def get_num_steps(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        return self.epochs_count[work]

    def generate(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        fd_feature = self.fd_feature[work]
        fd_label = self.fd_label[work]
        epochs = self.get_num_steps(work)
        while True:
            for _ in range(epochs):
                feature = np.load(fd_feature)
                label = np.load(fd_label)
                yield feature, label
            self._reset_file(work)
            fd_feature = self.fd_feature[work]
            fd_label = self.fd_label[work]


class DataGeneratorMock:
    def __init__(self, encoder, batch, epochs_train=6, epochs_test=2):
        self.encoder = encoder
        self.batch = batch
        self.epochs = {'train': epochs_train, 'test': epochs_test}

    def get_num_steps(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        return self.epochs[work]

    def generate(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        while True:
            feature = np.random.rand(*((self.batch, ) + self.encoder.shape()))
            label = np.random.rand(*(self.batch, self.encoder.num_moves()))
            yield feature, label


if __name__ == '__main__':
    # Test code
    multiprocessing.freeze_support()
    generator = DataGenerator(TwoPlaneEncoder(5), 256, '../../dataset', '../encoded_data', 0.5)
    generator = DataGeneratorMock(TwoPlaneEncoder(5), 256)
    i = 0
    for feature, label in generator.generate('train'):
        print(feature.shape)
        print(label.shape)
        i += 1
        if i > 10:
            break
