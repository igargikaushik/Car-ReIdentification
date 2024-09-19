import os
import cv2
import numpy as np
import identification_color
import identify_feature
import read_mat_file
import sys

file_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/'
file_path1 = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID/'
def produce_color_txt():
    dirs = os.listdir(file_path)
    for i in dirs:
        img_path_class = os.path.join(file_path,i)
        dirs_class_car = os.listdir(img_path_class)
        print(dirs_class_car)
        for j in dirs_class_car:
            print(img_path_class,j)
            img_path_diff = os.path.join(img_path_class, j)
            dirs_img_path = os.listdir(img_path_diff)
            for k in dirs_img_path:
                img_path = os.path.join(img_path_diff,k)
                img = cv2.imread(img_path)
                if os.path.splitext(k)[1] == '.jpg':
                    path = img_path_diff+'/'+os.path.splitext(k)[0]+'.txt'
                    print(path)
                    f = open(img_path_diff+'/'+os.path.splitext(k)[0]+'.txt','w')
                    color = identification_color.recog_car_color(img)
                    for z in color:
                        f.write(z)
                        f.write('\n')
                    f.write(j)
                    f.close()

     


def classify_img():
    f = open(file_path1+'query.txt','w')
    f1 = open(file_path1+'dataset.txt','w')
    dirs = os.listdir(file_path)
    for i in dirs:
        img_path_class = os.path.join(file_path,i)
        dirs_class_car = os.listdir(img_path_class)
        for j in dirs_class_car:
            img_path_diff = os.path.join(img_path_class, j)
            dirs_img_path = os.listdir(img_path_diff)
            for k in dirs_img_path:
                img_path = os.path.join(img_path_diff,k)
                if os.path.splitext(k)[1] == '.jpg':
                    f1.write(img_path)
                    f1.write('\n')
                    print(img_path)
                    break
    cnt_lisense = 0
    for i in dirs:
        img_path_class = os.path.join(file_path,i)
        dirs_class_car = os.listdir(img_path_class)
        for j in dirs_class_car:
            img_path_diff = os.path.join(img_path_class, j)
            dirs_img_path = os.listdir(img_path_diff)
            cnt_img = 0
            for k in dirs_img_path:
                img_path = os.path.join(img_path_diff,k)
                if cnt_lisense == 100:
                    f1.close()
                    f.close()
                    return 0
                if os.path.splitext(k)[1] == '.jpg':
                    cnt_img+=1
                    if cnt_img==2:
                        cnt_lisense += 1
                        f.write(img_path)
                        f.write('\n')
                        break

def isInter(a,b):
		result = list(set(a)&set(b))
		if result:
			return True
		else:
			return False




def select_color_img(query_img,data_list):
    f = open(os.path.splitext(query_img)[0]+".txt",'r')
    return_data_list = []
    query_color = f.read().split('\n')
    query_color.pop()
    for i in data_list:
        f1 = open(os.path.splitext(i)[0]+".txt",'r')
        data_color = f1.read().split('\n')
        data_color.pop()
        if isInter(query_color,data_color):
            return_data_list.append(i)
    return  return_data_list


def test_color_code(query_list,data_list):
    query_nums = len(query_list)

    for i in data_list:
        pass


def produce_feature_data(data_list):
    if os.path.splitext(data_list[0])[0].endswith("npy"):
        return
    for j,i in enumerate(data_list):
        print("第",j,"总共",len(data_list))
        img = cv2.imread(i)
        keypoints, descriptors =  cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
        np.save( os.path.splitext(i)[0]+".npy", descriptors)

def produce_feature_windows_data(data_list):
    if os.path.splitext(data_list[0])[0].endswith("npy"):
        return
    for j,i in enumerate(data_list):
        print("第",j,"总共",len(data_list))
        img = (read_mat_file.get_feature_roi(i))
        keypoints, descriptors =  cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
        np.save( os.path.splitext(i)[0]+"_w.npy", descriptors)

def run():
    f = open(file_path1+'query.txt','r')
    f1 = open(file_path1 + 'dataset.txt', 'r')
    list_query = f.read().split('\n')
    list_query.pop()
    list_dataset = f1.read().split('\n')
    list_dataset.pop()

    cnt_right = 0
    j = 0

    for j,i in enumerate(list_query):
        color_data_list = select_color_img(i,list_dataset)
        flag = identify_feature.query(i,color_data_list)
        if flag:
            print("第", j, "结果预测正确")
            cnt_right+=1
        else:
            print("第", j, "结果预测错误")
        sys.stdout.write("\r data: %d / %d" % (cnt_right, j))
    print(cnt_right)





if __name__ == '__main__':
    
    run()