import os
import re

def get_checkpoint(path = "./" , reg = r"checkpoint-(\d+).h5$"):
    regex = re.compile(reg)
    checkpoint_name = None
    max_epoch = 0
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        if os.path.isfile(file_name):
            fit_list = regex.findall(file)
            if len(fit_list) == 0:
                continue 
            current_epoch = int(fit_list[-1])
            if current_epoch > max_epoch:
                max_epoch = current_epoch
                checkpoint_name = file_name
    return checkpoint_name, max_epoch

if __name__=='__main__':
    print(get_checkpoint(".\\version-1"))