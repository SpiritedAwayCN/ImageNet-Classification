import json
import os

with open('imagenet_class_index.json') as f:
    label_dict = json.loads(f.readline())
    write_dict = {}
    i = 0
    cnt = 0
    for dir_name in os.listdir("./train"):
        # print(dir_name)
        while label_dict[str(i)][0] != dir_name:
            i += 1
        
        write_dict.update({str(cnt): label_dict[str(i)][1]})
        cnt += 1

with open('labels_to_contents.json', "w") as f:
    json.dump(write_dict, f)