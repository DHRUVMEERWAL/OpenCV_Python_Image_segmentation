# OpenCV_Python_Image_segmentation

Image Segmentation with feature detection (Burnt and Unburnt Area ) using OpenCV-Python and Numpy

The theme for this task is Search and Rescue. A fire has broken out in a civilian area and your
job is to gather information about the location of houses and buildings in the area. Your UAV is
collecting images of the search area that look like the sample image given below.
Information about the input image

● The brown area is burnt grass
● The green area is unburnt (green) grass
● The blue and red triangles are houses
● Each house colour has an associated priority
● Blue house have a priority of 2
● Red houses have a priority of 1

**Input**

A list of 10 images, sample file with the code is given.

**Expected Output**

1. An output image, for each input image, that clearly shows the difference between the
burnt grass and green grass, by overlaying 2 unique colors on top of each. The expected
output for the given sample input is given below
2. The number of houses on the burnt grass (Hb) and the number of houses on the green
grass (Hg), saved in a list
3. The total priority of houses on the burnt grass (Pb) and the total priority of houses on the
green grass (Pg), saved in a list
4. A rescue ratio of priority Pr where Pr = Pb
/ Pg
, saved in a list
5. A list of the names of the input images , arranges in descending order of their rescue ratio
(Pr)

##Explanation

The code comprises of a main() function which fetches info from other functions nested within it .
Main() calls 4 defined function mainly :
A). #STEP 1: Calulating All Blue Houses(Burnt + Unburnt)
    Count_Blue_House(Image,Blue_Houses,Blue_House_Img)

B). #STEP 2: Calulating All Red Houses(Burnt + Unburnt)
    Count_Red_House(Image,Red_Houses)

C). #STEP 3: Caluating All Unburnt Houses(Blue + Red)
    Count_Unburnt_House(Image,Unburnt_Houses)
    
D).  #STEP 4: Calculating Unburnt Red House
    Count_Unburnt_Red_House(Image,Unburnt_Red_Houses)


Approch to the problem is to first Count All Blue,Red,Unburnt and Unburnt Red House
as :

Burnt Houses = Burnt Blue House + Burnt Red Houses
Unburnt Houses = Unburnt Blue House + Unburnt Red Houses

So, using the above equations we can calculate the given variables.

Now, 
