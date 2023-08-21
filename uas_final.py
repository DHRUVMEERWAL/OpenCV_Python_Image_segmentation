import cv2
import numpy as np


def Count_Blue_House(Image):
    import cv2
    import numpy as np
    img = cv2.imread(Image)

    lower_range = (00, 00, 00) # lower range of blue color in BGR
    upper_range = (255, 00, 00) # upper range of blue color in BGR
    mask = cv2.inRange(img, lower_range, upper_range)
    color_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite('Count_Blue_House_1.png',color_image)

    img = cv2.imread('Count_Blue_House_1.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 0, 0])
    origMask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(origMask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    cv2.imwrite('Count_Blue_House_2.png',mask)


    image = cv2.imread('Count_Blue_House_2.png')


    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)


    # Finding Contours

    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
    	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    global Blue_Houses
    Blue_Houses=int(str(len(contours)))



def Red_House(Image):
    import cv2
    import numpy as np
    img = cv2.imread(Image)

    lower_range = (00, 00, 00) # lower range of red color in BGR
    upper_range = (00, 00, 255) # upper range of red color in BGR
    mask = cv2.inRange(img, lower_range, upper_range)
    color_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite('Count_Red_House_1.png',color_image)

    img = cv2.imread('Count_Red_House_1.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 20, 20])
    upper_bound = np.array([20, 255, 255])
    origMask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(origMask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    cv2.imwrite('Count_Red_House_2.png',mask)
    

    img = cv2.imread('Count_Red_House_2.png')

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    #cv2.waitKey(0)

    # Finding Contours

    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    global Red_Houses
    Red_Houses=int(str(len(contours)))


def Unburnt_House(Image):


    import cv2
    import numpy as np
    img = cv2.imread(Image)

    lower_range = (0, 50, 0) # lower range of green color in BGR
    upper_range = (15, 255,65 ) # upper range of green color in BGR
    mask = cv2.inRange(img, lower_range, upper_range)


    color_image = cv2.bitwise_and(img, img, mask=mask)

    color_image[np.where((color_image==[0,0,0]).all(axis=2))]=[225,225,0]


    cv2.imwrite('Unburnt_Area_1.jpg',color_image)


    image = cv2.imread('Unburnt_Area_1.jpg')
    th, im_th = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), (255,0,0))



    cv2.imwrite('Unburnt_Area_2.png', im_floodfill)



    img = cv2.imread('Unburnt_Area_2.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 20, 20])
    upper_bound = np.array([20, 255, 255])
    origMask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(origMask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    cv2.imwrite('Unburnt_Area_3.png',mask)



    image = cv2.imread('Unburnt_Area_3.png')

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    #cv2.waitKey(0)

    # Finding Contours

    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    global Unburnt_Houses
    Unburnt_Houses=int(str(len(contours)))



def Unburnt_Red_House(Image):


    import cv2
    import numpy as np
    img = cv2.imread(Image)


    lower_range = (0, 50, 0) # lower range of green color in BGR
    upper_range = (15, 255,65 ) # upper range of green color in BGR
    mask = cv2.inRange(img, lower_range, upper_range)


    color_image = cv2.bitwise_and(img, img, mask=mask)

    color_image[np.where((color_image==[0,0,0]).all(axis=2))]=[225,225,0]


    cv2.imwrite('Uburnt_Red_House_1.jpg',color_image)



    image = cv2.imread('Uburnt_Red_House_1.jpg')


    th, im_th = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), (255,0,0))

    cv2.imwrite('Uburnt_Red_House_2.png', im_floodfill)

    # Load two images
    img1 = cv2.imread('Uburnt_Red_House_2.png')

    img2 = cv2.imread('Count_Blue_House_1.png')


    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put Houses in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst

    cv2.imwrite('Uburnt_Red_House_3.png',img1)



    img = cv2.imread('Uburnt_Red_House_3.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 20, 20])
    upper_bound = np.array([20, 255, 255])
    origMask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(origMask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    cv2.imwrite('Uburnt_Red_House_4.png',mask)



    image = cv2.imread('Uburnt_Red_House_4.png')

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
    #cv2.waitKey(0)

    # Finding Contours

    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    global Unburnt_Red_Houses

    Unburnt_Red_Houses=int(str(len(contours)))



def main():
    Image_List=[]
    Number_Image=int(input('Number of Images need to be proccessed : '))
    for i in range(Number_Image):
        Path=input('Enter the Image Name or Image Path with Extension : ')
        Image_List.append(Path)
    #print(Image_List)

    #Defining Few Lists
    INFO_ALL_IMAGES_LIST=[]
    n_house_list=[]
    priority_houses_list=[]
    priority_ratio_list=[]
    image_by_rescue_ratio_list=[]
    CURRENT_INFO_LIST=[]
    
    
        
    for j in range(len(Image_List)):

        #Defining Few Variables

        '''Blue_Houses=0
        Red_Houses=0
        
        Unburnt_Houses=0
        Unburnt_Red_Houses=0
        Unburnt_Blue_Houses=0
        
        Burnt_Houses=0
        Burnt_Red_Houses=0
        Burnt_Blue_Houses=0
        
        Total_Houses=int(Blue_Houses)+int(Red_Houses)
        Total_Houses=0'''
        Rescue_Ratio_Priority = 0

        Blue_House_Img = None
        
        Image=Image_List[j]

        #STEP 1: Calulating All Blue Houses(Burnt + Unburnt)
        Count_Blue_House(Image)

        #STEP 2: Calulating All Red Houses(Burnt + Unburnt)
        Red_House(Image)

        #STEP 3: Caluating All Unburnt Houses(Blue + Red)
        Unburnt_House(Image)

        #STEP 4: Calculating Unburnt Red House
        Unburnt_Red_House(Image)

        Total_Houses=int(Blue_Houses)+int(Red_Houses)
        #Few Calculated Values
        
        Unburnt_Blue_Houses=Unburnt_Houses - Unburnt_Red_Houses
        #print(Unburnt_Blue_Houses)
        Burnt_Houses=Total_Houses - (Unburnt_Red_Houses+Unburnt_Blue_Houses)
        #print(Burnt_Houses)
        Burnt_Red_Houses=Red_Houses - Unburnt_Red_Houses
        #print(Burnt_Red_Houses)
        Burnt_Blue_Houses=Blue_Houses - Unburnt_Blue_Houses
        #print(Burnt_Blue_Houses)
        rescue_ratio=((1*Burnt_Red_Houses)+(2*Burnt_Blue_Houses))/((1*Unburnt_Red_Houses)+(2*Unburnt_Blue_Houses))

        CURRENT_INFO_LIST.append(j+1)
        CURRENT_INFO_LIST.append('Image'+str(j+1))
        CURRENT_INFO_LIST.append(Burnt_Houses)
        CURRENT_INFO_LIST.append(Unburnt_Houses)
        CURRENT_INFO_LIST.append((1*Burnt_Red_Houses)+(2*Burnt_Blue_Houses))
        CURRENT_INFO_LIST.append((1*Unburnt_Red_Houses)+(2*Unburnt_Blue_Houses))
        CURRENT_INFO_LIST.append(rescue_ratio)

        #print(CURRENT_INFO_LIST)
        
        #INFO_ALL_IMAGES_LIST.append(CURRENT_INFO_LIST)
        #print(INFO_ALL_IMAGES_LIST)

        #CURRENT_INFO_LIST.clear()
        #rescue_ratio=0
        #CURRENT_INFO_LIST INDEXING
        #0: NUMBER
        #1: IMAGE_NAME
        #2: HOUSES_IN_BURNT_AREA
        #3: HOUSES_IN_UNBURNT_AREA
        #4: PRIORITY_BURNT_AREA
        #5: PRIORITY_UNBURNT_AREA
        #6: PRIORITY_RATIO

    #Calculting_list=[]
    #print(CURRENT_INFO_LIST)
    #print(INFO_ALL_IMAGES_LIST)
    for k in range(len(Image_List)):
        n_house_list.append(CURRENT_INFO_LIST[2+(k*7):4+(k*7)])
        priority_houses_list.append(CURRENT_INFO_LIST[4+(k*7):6+(k*7)])
        priority_ratio_list.append(CURRENT_INFO_LIST[6+(k*7)])
        image_by_rescue_ratio_list.append(CURRENT_INFO_LIST[1+(k*7)])
    image_by_rescue_ratio_list.sort(reverse=True)
    print('n_houses = ',n_house_list)
    print('priority_houses = ',priority_houses_list)
    print('priority_ratio = ',priority_ratio_list)
    print('image_by_rescue_ratio = ',image_by_rescue_ratio_list)

        
main()




