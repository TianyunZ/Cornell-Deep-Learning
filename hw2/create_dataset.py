import os
import sys
import shutil

pet_dict = {}
for file in os.listdir("images"):
    pet_list = file.split('.')[0].split('_')
    pet_name = " ".join(pet_list[:-1]).lower()
    new_file = "E:/Tianyun/Cornell_Tech/Courses/21SP/DL/hw2/" 
    if pet_name not in pet_dict.keys():
        pet_dict[pet_name] = 1
        new_file += "oxford_pet_37/train/"+pet_name+"/"
        if not os.path.exists("oxford_pet_37/train/"+pet_name):
            os.makedirs("oxford_pet_37/train/"+pet_name)
    else:
        if pet_dict[pet_name] > 150:
            new_file += "oxford_pet_37/test/"+pet_name+"/"
            if not os.path.exists(new_file):
                os.makedirs(new_file)
        else:
            new_file += "oxford_pet_37/train/"+pet_name+"/"
        pet_dict[pet_name] += 1
    shutil.copy("images/"+file, new_file+file)
    # print(new_file+file)
    # sys.exit()