# 上交标注数据 to Yolo

import os
from pathlib import Path
out_path = './labels'
if(os.path.exists(out_path) == False):
    os.mkdir(out_path)
src_label_path = './sj_labels' 
files= os.listdir(src_label_path) 

for file in files: 
     if not os.path.isdir(file): 
            s = []
            f = open(src_label_path+"/"+file); 
            iter_f = iter(f); 
            for line in iter_f:
                string = line.split(' ')
                cls = string[0]
                data = [float(i) for i in string[1:]]
                left = 1
                up  = 1
                right = 0
                down = 0 

                for i in range(0,8):
                    if(i % 2 == 0):
                        if((data[i]) < left):
                            left = data[i]

                        if(data[i] > right):
                            right = data[i]

                    if(i % 2 == 1):
                        if(data[i] < up):
                            up = data[i]

                        if(data[i] > down):
                            down = data[i]

                width = right - left 
                height = down - up

                center_x = (right + left) / 2
                center_y = (down + up) / 2      
                
                res = str(cls)  + ' ' + str(center_x)  + ' ' + str(center_y) + ' ' +  str(width) +' ' +  str(height) + '\n'
                s.append(res)
            with open(out_path + '/' + file,'w+') as f_dst:
                f_dst.writelines(s)