import os
import multiprocessing
import queue
import struct
import time
import glob

from dlabalone.encoders.twoplane import TwoPlaneEncoder
from dlabalone.utils import load_file_board_move_pair
from tensorflow.keras.utils import to_categorical
import numpy as np


def _convert_dataset_to_npz(encoder, batch_size, file_list, out_filename):
    thread_count = os.cpu_count() - 1

    # Put file names & Start threads to encode data
    q_infile = multiprocessing.Queue()
    q_ggodari = multiprocessing.Queue()

    for file in file_list:
        q_infile.put_nowait(file)

    # Get encoded data and do work
    index = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    # with multiprocessing.Pool(processes=thread_count) as p:
    #     result_list = p.starmap(encode_data_worker, (encoder, batch_size, out_filename, q_infile, q_data, index, lock))
    thread_list = []
    for _ in range(thread_count):
        p = multiprocessing.Process(target=encoder.encode_data_worker,
                                    args=(encoder, batch_size, out_filename, q_infile, q_ggodari, index, lock))
        p.daemon = False
        p.start()
        thread_list.append(p)

    ggodari_list = []
    for _ in range(thread_count):
        ggodari_list.append(q_ggodari.get())
    for p in thread_list:
        p.join()

    # Save ggodari
    encoder.ggodari_merger(ggodari_list, batch_size, out_filename, index.value)


    # feature_list = []
    # label_list = []
    # for _ in range(thread_count):
    #     feature_remains, label_remains = q_ggodari.get()
    #
    # for p in thread_list:
    #     p.join()
    #
    # # Save ggodari
    # print('Start to save ggodari')
    # while len(feature_list) > 0:
    #     features = np.concatenate(feature_list[:batch_size], axis=0)
    #     labels = np.concatenate(label_list[:batch_size], axis=0)
    #     feature_list = feature_list[batch_size:]
    #     label_list = label_list[batch_size:]
    #     np.savez_compressed(out_filename % index.value, feature=features, label=labels)
    #     print(out_filename % index.value)
    #     index.value += 1


def convert_dataset_to_npz(encoder, batch, in_dir, out_filename_format, test_ratio, overwrite=False):
    out_filename_train = out_filename_format % ('train', '%s')
    out_filename_test = out_filename_format % ('test', '%s')
    dirname = os.path.dirname(out_filename_train)
    if overwrite or (not os.path.isdir(dirname)):
        # Count file
        data_files = [f for f in map(lambda x: os.path.join(in_dir, x), os.listdir(in_dir)) if os.path.isfile(f)]
        file_count = len(data_files)
        train_file_count = int(file_count * (1 - test_ratio))

        # Separate file for train and test
        train_files = data_files[:train_file_count]
        test_files = data_files[train_file_count:]

        print('Converting train data to npz...')
        os.mkdir(dirname)
        _convert_dataset_to_npz(encoder, batch, train_files, out_filename_train)
        print('Converting train data to npz done.')
        print('Converting test data to npz...')
        _convert_dataset_to_npz(encoder, batch, test_files, out_filename_test)
        print('Converting test data to npz done.')
    else:
        print('npz already exists. Skipping converting')


def npz_preloader(encoder, q_infile, q_data):
    while True:
        q_data.put(encoder.load(q_infile.get()))


class DataGenerator:
    def __init__(self, encoder, batch, dataset_dir, out_dir, test_ratio, out_filename_format=None, overwrite=False,
                 max_train_npz=None):
        self.encoder = encoder
        multiprocessing.freeze_support()
        if out_filename_format is None:
            out_filename_format = f'{encoder.name()}_{batch}batch_{test_ratio}/%s_%s.npz'
        out_filename_format = os.path.join(out_dir, out_filename_format)
        convert_dataset_to_npz(encoder, batch, dataset_dir, out_filename_format, test_ratio, overwrite)
        self.out_filename_format = out_filename_format

        # Count the number of files == step count
        dirname = os.path.dirname(out_filename_format)
        self.npz_files = {'test': glob.glob(dirname + '/test_*.npz'), 'train': glob.glob(dirname + '/train_*.npz')}
        if max_train_npz is not None:
            self.npz_files['train'] = self.npz_files['train'][:max_train_npz]
            max_test_npz = int(max_train_npz * test_ratio / (1 - test_ratio))
            self.npz_files['test'] = self.npz_files['test'][:max_test_npz]
        self.step_count = {'test': len(self.npz_files['test']), 'train': len(self.npz_files['train'])}

    def get_num_steps(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        return self.step_count[work]

    def generate(self, work, use_multithread=False):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        if use_multithread:
            thread_list = []
            thread_count = 4
            q_infile = multiprocessing.Queue(maxsize=len(self.npz_files[work]) * 2)
            q_data = multiprocessing.Queue(maxsize=thread_count * 16)
            for _ in range(thread_count):
                p = multiprocessing.Process(target=npz_preloader, args=(self.encoder, q_infile, q_data))
                p.daemon = True
                p.start()
                thread_list.append(p)
        else:
            thread_list = []
            q_infile = None
            q_data = None
        try:
            while True:
                if use_multithread:
                    for file in self.npz_files[work]:
                        q_infile.put(file)
                    for _ in range(len(self.npz_files[work])):
                        feature, label = q_data.get()
                        yield feature, label
                else:
                    for file in self.npz_files[work]:
                        feature, label = self.encoder.load(file)
                        yield feature, label
        except GeneratorExit:
            if use_multithread:
                for p in thread_list:
                    p.terminate()
                    p.join()


class DataGeneratorMock:
    def __init__(self, encoder, batch, steps_train=6, steps_test=2):
        self.encoder = encoder
        self.batch = batch
        self.steps = {'train': steps_train, 'test': steps_test}
        self.feature = np.random.rand(*((self.batch,) + self.encoder.shape()))
        self.label = np.random.rand(*((self.batch,) + self.encoder.label_shape()))
        # self.label = np.concatenate([
        #     np.ones(((self.batch // 2,) + self.encoder.label_shape())),
        #     np.ones(((self.batch // 2,) + self.encoder.label_shape())) * -1,])
        # print(self.label)

    def get_num_steps(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        return self.steps[work]

    def generate(self, work):
        assert work in ['train', 'test'], f'Invalid work type {work}'
        while True:
            # feature = np.random.rand(*((self.batch, ) + self.encoder.shape()))
            # label = np.random.rand(*(self.batch, self.encoder.num_moves()))
            yield self.feature, self.label


if __name__ == '__main__':
    # Test code
    multiprocessing.freeze_support()
    generator = DataGenerator(TwoPlaneEncoder(5), 256, '../../data/dataset', '../../data/encoded_data', 0.5)
    # generator = DataGeneratorMock(TwoPlaneEncoder(5), 256)
    # i = 0
    # for feature, label in generator.generate('train'):
    #     print(feature.shape)
    #     print(label.shape)
    #     i += 1
    #     if i > 10:
    #         break
