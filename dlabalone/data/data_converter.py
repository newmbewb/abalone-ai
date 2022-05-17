import os


def plaindata2valuedata(dir_plaindata, dir_valuedata, file_filter=None):
    if file_filter is None:
        def filter_default(path):
            if 'draw' in path:
                return False
            return True
        file_filter = filter_default
    for f in os.listdir(dir_plaindata):
        path_plain = os.path.join(dir_plaindata, f)
        if not file_filter(path_plain):
            continue
        path_value = os.path.join(dir_valuedata, f)
        fd_plain = open(path_plain, 'r')
        fd_value = open(path_value, 'w')
        first_line = fd_plain.readline()
        board_size, line_count = first_line.split(',')
        line_count = int(line_count)
        if line_count % 2 == 1:
            next_value_sign = 1
        else:
            next_value_sign = -1
        fd_value.write(first_line)
        step = 0
        for line in fd_plain:
            step += 1
            value_size = step / line_count
            line = line.strip()
            line += f'&1&{next_value_sign * value_size}'
            next_value_sign *= -1
            fd_value.write(line + '\n')
        fd_plain.close()
        fd_value.close()
