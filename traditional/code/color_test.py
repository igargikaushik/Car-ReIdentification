import os
import cv2
import identification_color

img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/3\License_203/'

str = []
str1 = {'blank': 0, 'gray': 0, 'white': 0, 'red': 0,
              'origin': 0, 'yellow': 0, 'green': 0, 'qing': 0,
              'blue': 0,
              'purple': 0}
dirs = os.listdir(img_path)
for dir in dirs:
    if(os.path.splitext(dir)[1] == '.jpg' or os.path.splitext(dir)[1] == '.png'):
        str.append(dir)

def test_img():
    for i in str:
        
        f = open(img_path+os.path.splitext(i)[0]+"color"+".txt",'w')
        img = cv2.imread(img_path+i)
        color = identification_color.recog_car_color(img)
        for j in color:
            f.write(j)
            f.write('\n')
        f.close()
        f = open(img_path + os.path.splitext(i)[0] + "color" + ".txt", 'r')
        s = f.read().split('\n')
        s.pop()
        print("s",s)
        f.close()
       
    print(str1)