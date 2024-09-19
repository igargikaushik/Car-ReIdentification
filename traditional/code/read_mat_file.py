import scipy.io as scio
import cv2
import os

path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID/sv_window_loc.mat'


def get_car_windows(mat_path):
    data = scio.loadmat(path)
    window_data = data['sv_window_loc']
    window_dict = {}
    for w in window_data:
        data = []
        for i in range(1,5):
            data.append(int(w[i][0]))

        window_dict[w[0][0]] = data
       

    return window_dict

def test(d):
    windows_dict = d
    input_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/1\License_1/'
    output_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\output_windows_img\dataset/'
    dirs_car = os.listdir(input_path)
    for k in dirs_car:
        if os.path.splitext(k)[1] == '.jpg':
            img_path = input_path+k
            img = cv2.imread(img_path)
            img_roi = img[(windows_dict[k][1]):(windows_dict[k][3]),(windows_dict[k][0]):(windows_dict[k][2]-int((windows_dict[k][2] - windows_dict[k][0])/3*2))]
            cv2.imwrite(output_path+k,img_roi)

def get_feature_roi(img_path):
    windows_dict = get_car_windows(path)
    print(img_path)
    img = cv2.imread(img_path)
    k = img_path.split('\\')[-1]
    img_roi = img[(windows_dict[k][1]):(windows_dict[k][3]),
              (windows_dict[k][0]):(windows_dict[k][2] - int((windows_dict[k][2] - windows_dict[k][0]) / 3 * 2))]
    return img_roi

if __name__ == '__main__':
    windows_dict = get_car_windows(path)
    test(windows_dict)