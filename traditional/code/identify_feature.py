import os
import sys
import numpy as np
import cv2
import read_mat_file
img_path = 'G:/19\lab\python\Vehicle Re-identification/feature\code\query/4404000000002948081100.jpg'
query = cv2.imread(img_path, 0)
folder = './data'
descriptors = []

data_distrib_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID/'

index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def query(query_img_name,data_list):
    query_img = cv2.imread(query_img_name)
    sift = cv2.xfeatures2d.SIFT_create()
    query_kp, query_ds = sift.detectAndCompute(query_img, None)
    potential_culprits = {}
    le = len(data_list)
    for j,i in enumerate(data_list):
        sys.stdout.write("\n data: %d / %d" % (j, le))
        matches = flann.knnMatch(query_ds, np.load(os.path.splitext(i)[0]+".npy"), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        potential_culprits[i] = len(good)
       
    query_img = (read_mat_file.get_feature_roi(query_img_name))
    sift = cv2.xfeatures2d.SIFT_create()
    query_kp, query_ds = sift.detectAndCompute(query_img, None)
    for j,i in enumerate(data_list):
        sys.stdout.write("\n data: %d / %d" % (j, le))
        matches = flann.knnMatch(query_ds, np.load(os.path.splitext(i)[0]+"_w.npy"), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        potential_culprits[i] += 2*len(good)

    sort_list = sorted(potential_culprits.items(), key=lambda item: item[1], reverse=True)
    print(sort_list)
    flag = False
    for i in range(le):
        car_info = sort_list[i][0]
        f = open(os.path.splitext(car_info)[0]+".txt","r")
        f1 = open(os.path.splitext(query_img_name)[0]+".txt","r")
        car_data = f.read().split('\n')
        car_query_data = f1.read().split('\n')
        if car_data[2] == car_query_data[2]:
            with open(data_distrib_path+"data_distrib.txt","a+") as f_d:
                f_d.write(str(i))
                f_d.write('\n')
                flag = True
            if i < 100:
                return True
    if flag == False:
        with open(data_distrib_path + "data_distrib.txt", "a+") as f_d:
            f_d.write(str(999))
            f_d.write('\n')
    return False


   