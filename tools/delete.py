# Delete Invilde Data
import os
img_dir = './images'
lable_dir = './labels'

labels = os.listdir(lable_dir)
files= os.listdir(img_dir)

for img_file in  files:
    img_n = img_file.split('.')
    if (not str(img_n[0]) + '.txt' in labels):
       os.remove(img_dir +'/' + img_file)