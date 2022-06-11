import re

if __name__ == '__main__':
    logname = 'log/rl_mcts/gen01_policy_dropout0.3_old+new_acsimple1.txt'

    p1 = re.compile(
        r'.*val_loss: (?P<val_loss>[0-9\.]+) - '
        r'val_accuracy: (?P<val_accuracy>[0-9\.]+) - '
        r'val_top3_acc: (?P<val_top3_acc>[0-9\.]+) - '
        r'val_top5_acc: (?P<val_top5_acc>[0-9\.]+)'
    )
    p2 = re.compile(
        r'.*loss: (?P<loss>[0-9\.]+) - '
        r'mean_squared_error: (?P<mean_squared_error>[0-9\.]+) - '
        r'val_loss: (?P<val_loss>[0-9\.]+) - '
        r'val_mean_squared_error: (?P<val_mean_squared_error>[0-9\.]+)')
    fd = open(logname, 'r')
    max_accuracy = 0
    max_m = None
    max_epoch = 0
    epoch = 0
    for line in fd:
        m = p1.match(line)
        if m:
            epoch += 1
            accuracy = float(m.group('val_accuracy'))
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                max_m = m
                max_epoch = epoch
        # m = p2.match(line)
        # if m:
        #     val_metric.append(m.group('val_mean_squared_error'))
        #     metric.append(m.group('mean_squared_error'))
    print(f"{max_m.group('val_accuracy')},{max_m.group('val_top3_acc')},{max_m.group('val_top5_acc')},{max_epoch}")
