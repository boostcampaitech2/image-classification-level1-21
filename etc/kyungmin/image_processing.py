import os

os.system('mkdir ./train')
os.system('mkdir ./train/new_images')

mask_list = ['mask', 'incorrect', 'normal']
gender_list = ['male', 'female']
age_list = ['under', 'middle', 'upper']

cnt = 0
for mask in mask_list:
    for gender in gender_list:
        for age in age_list:
            if cnt < 10:
                num = "0" + str(cnt)
            else:
                num = str(cnt)
            
            label_folder = "{}_{}_{}_{}".format(num, mask, gender, age)
            os.system('mkdir ./train/new_images/' + label_folder)
            cnt += 1

BASE_PATH = './train/face_boxes/'
folder_list = [f for f in os.listdir(BASE_PATH) if not f.startswith('.')] 

for folder in folder_list:
    string = folder.split('_')
    if string[0] == '.':
        continue
    #if len(string) < 3:
    #    continue
    id = string[0]
    gender = string[1]
    age = string[3]

    if gender == 'male':
        gender = gender_list[0]
    elif gender == 'female':
        gender = gender_list[1]

    if int(age) < 26:
        age = age_list[0]
    elif int(age) < 53:
        age = age_list[1]
    else:
        age = age_list[2]
    
    image_path = BASE_PATH + folder +'/'
    file_list = [f for f in os.listdir(image_path) if not f.startswith('.')]

    for file in file_list:
        if file.find(mask_list[0]) == 0:
            mask = mask_list[0]
        elif file.find(mask_list[1]) == 0:
            mask = mask_list[1]
        elif file.find(mask_list[2]) == 0:
            mask = mask_list[2]
        
        label_list = os.listdir('./train/new_images')

        for label in label_list:
            if label.find("{}_{}_{}".format(mask,gender,age)) != -1:
                os.system('cp ' + image_path + file + ' ./train/new_images/' + label + '/' + id + file)        
