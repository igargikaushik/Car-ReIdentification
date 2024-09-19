import cv2
img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/1\License_6/4404000000002943547902.jpg'

img = cv2.imread(img_path)
def find_car_contour(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret, binary = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.dilate(binary,kernel)
    binary = cv2.erode(binary,kernel)
    binary = cv2.dilate(binary, kernel)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
       
        area = img.shape[0] * img.shape[1] #shape[0]--height
        if w*h > area/2:
            cv2.rectangle(img, (x, y+int(h/2)), (x + w, y + h), (153, 153, 233), 5)
           
            return (x, y, w, h)

    return 0,0,img.shape[1],img.shape[0]
if __name__ == '__main__':
   
  find_car_contour(img)
  cv2.imshow("img",img)
  cv2.waitKey()