import os
import shutil
import numpy as np

for dirs in os.listdir('train\\'):
    src = os.path.join('train', dirs)
    if not os.path.isdir(src):
        continue
    print('processing', dirs, '...')

    files = os.listdir(src)
    np.random.shuffle(files)

    dst = os.path.join('val\\', dirs)
    if not os.path.isdir(dirs):
        os.system('mkdir {}'.format(dst))
    dst = os.path.join(dst, "")

    for i in range(0, 50):
        shutil.move(os.path.join(src, files[i]), dst)
        # exit(0)
    
    dst = os.path.join('test\\', dirs)
    if not os.path.isdir(dirs):
        os.system('mkdir {}'.format(dst))
    dst = os.path.join(dst, "")
    for i in range(50, 100):
        shutil.move(os.path.join(src, files[i]), dst)
    