import re

if __name__ == '__main__':
    logname = 'log/gen01_value/acsimple6_dropout0.3_sgd0.1.txt'

    # p1 = re.compile(
    #     r'.*loss: (?P<loss>[0-9\.]+) - '
    #     r'accuracy: (?P<accuracy>[0-9\.]+) - '
    #     r'val_loss: (?P<val_loss>[0-9\.]+) - '
    #     r'val_accuracy: (?P<val_accuracy>[0-9\.]+)')
    # p2 = re.compile(
    #     r'.*loss: (?P<loss>[0-9\.]+) - '
    #     r'mean_squared_error: (?P<mean_squared_error>[0-9\.]+) - '
    #     r'val_loss: (?P<val_loss>[0-9\.]+) - '
    #     r'val_mean_squared_error: (?P<val_mean_squared_error>[0-9\.]+)')
    p1 = re.compile(
        r'.*loss: (?P<loss>[0-9\.]+) - '
        r'accuracy: (?P<accuracy>[0-9\.]+) - '
        r'top3_acc: (?P<top3_acc>[0-9\.]+) - '
        r'top5_acc: (?P<top5_acc>[0-9\.]+) - '
        r'val_loss: (?P<val_loss>[0-9\.]+) - '
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
    val_metric = []
    metric = []
    for line in fd:
        m = p1.match(line)
        if m:
            val_metric.append(m.group('val_accuracy'))
            metric.append(m.group('accuracy'))
        m = p2.match(line)
        if m:
            val_metric.append(m.group('val_mean_squared_error'))
            metric.append(m.group('mean_squared_error'))
    print('val_metric')
    print(','.join(val_metric))
    print('metric')
    print(','.join(metric))
