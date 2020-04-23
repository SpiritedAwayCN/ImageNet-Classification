import os

with open('test_label.txt', 'w') as f:
    label = 0
    for dirs in os.listdir('test\\'):
        src = os.path.join('test', dirs)
        if not os.path.isdir(src):
            continue
        files = os.listdir(src)
        for file in files:
            f.write("{} {:d}\n".format(os.path.join(dirs, file), label))
        label += 1
    