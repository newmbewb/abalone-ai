from dlabalone.ablboard import Move
import time

move_list_delimiter = '&'


def is_same_move_set(list_a, list_b):
    if len(list_a) != len(list_b):
        return False
    for move_b in list_b:
        same_elem = False
        for move_a in list_a:
            if move_a == move_b:
                same_elem = True
                list_a.remove(move_a)
                break
        if not same_elem:
            return False
    assert len(list_a) == 0, 'Wrong code'
    return True


##################################
# File I/O Functions
##################################
def load_file_move_list(filename):
    fd = open(filename, 'r')
    move_str_list = fd.read().split(move_list_delimiter)
    fd.close()
    move_list = []
    for move_str in move_str_list:
        move_list.append(Move.str_to_move(move_str))
    return move_list


def save_file_move_list(filename, move_list):
    fd = open(filename, 'w')
    move_str_list = move_list_delimiter.join(map(str, move_list))
    fd.write(move_str_list)
    fd.close()


##################################
# Profiling Functions
##################################
class Profiler(object):
    def __init__(self):
        self.profile_start = {}
        self.profile_end = {}

    def start(self, name):
        self.profile_start[name] = time.time()

    def end(self, name):
        self.profile_end[name] = time.time()

    def print(self, name):
        print('Runtime (%s): %3.05f' % (name, self.profile_end[name] - self.profile_start[name]))


profiler = Profiler()
