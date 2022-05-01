import re

if __name__ == '__main__':
    logname = 'log/simple1_threeplane_adam_dropout0.1.txt'

    p = re.compile(r'.*loss: (?P<loss>[0-9\.]+) - accuracy: (?P<accuracy>[0-9\.]+) - val_loss: (?P<val_loss>[0-9\.]+) - val_accuracy: (?P<val_accuracy>[0-9\.]+)')
    fd = open(logname, 'r')
    val_accuracy = []
    accuracy = []
    for line in fd:
        m = p.match(line)
        if m:
            val_accuracy.append(m.group('val_accuracy'))
            accuracy.append(m.group('accuracy'))
    print('val_accuracy')
    print(','.join(val_accuracy))
    print('accuracy')
    print(','.join(accuracy))
