from dlabalone.abltypes import Player
import random
import copy
from dlabalone.utils import profiler
from dlabalone.ablboard import Board


def test_copy(size=100000):
    # Generate test data
    data_list = []
    list_sample = [Player.black] * 14 + [Player.white] * 14 + [None] * 53
    for _ in range(size):
        random.shuffle(list_sample)
        data_list.append(list(list_sample))

    board = Board.set_size(5)
    dict_sample = Board.valid_grids
    data_dict = []
    for _ in range(size):
        random.shuffle(dict_sample)
        new = {}
        for i in range(14):
            new[dict_sample[i]] = Player.white
            new[dict_sample[i+14]] = Player.black

        data_dict.append(dict(new))

    # Test
    profiler.start('list_deepcopy')
    for i in range(size):
        a = copy.deepcopy(data_list[i])
    profiler.end('list_deepcopy')
    profiler.print('list_deepcopy')

    profiler.start('list_copy')
    for i in range(size):
        a = copy.copy(data_list[i])
    profiler.end('list_copy')
    profiler.print('list_copy')

    profiler.start('list_list')
    for i in range(size):
        a = list(data_list[i])
    profiler.end('list_list')
    profiler.print('list_list')

    profiler.start('dict_deepcopy')
    for i in range(size):
        a = copy.deepcopy(data_dict[i])
    profiler.end('dict_deepcopy')
    profiler.print('dict_deepcopy')

    profiler.start('dict_dict')
    for i in range(size):
        a = dict(data_dict[i])
    profiler.end('dict_dict')
    profiler.print('dict_dict')


if __name__ == '__main__':
    test_copy(size=100000)
