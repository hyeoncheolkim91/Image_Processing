import cv2
import numpy as np
import sys
import math

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

rows, cols, bands = inputImage.shape 
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

tmp = np.copy(inputImage)
y_min = 0
y_max = 1
arr = np.array([], dtype=float)

for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        b, g, r = inputImage[i, j]
        gray = round(0.3*r + 0.6*g + 0.1*b + 0.5)
        

        #Convert to Non-linear rgb
        r=r/255
        g=g/255
        b=b/255

        #inv-gamma correction
        if(r<0.03928):
        	r=r/12.92
        else:
        	r=(np.power(((r+0.055)/1.055),2.4))

        if(g<0.03928):
        	g=g/12.92
        else:
        	g=(np.power(((g+0.055)/1.055),2.4))

        if(b<0.03928):
        	b=b/12.92
        else:
        	b=(np.power(((b+0.055)/1.055),2.4))
        
        rgb=[r,g,b]
   
        #Linear rgb  to XYZ
        matrix = [[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]]
        xyz=[]
        xyz= np.dot(matrix,rgb)
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]

        
        #XYZ to xyY
        if((x+y+z)!=0):
            x2= float(x/(x+y+z))
            y2= float(y/(x+y+z))
            z2= float(y)
        else:
            x2=0
            y2=0
            z2=0
    
        #append y value into array for max y and min y.
        arr = np.append(arr, y2)
        
        tmp[i, j] = [x2,y2,z2]

#Max pixel value in the histogram
mx = np.amax(arr)
#Min pixel valu in the hitogram
mn = np.amin(arr)



outputImage = np.copy(inputImage)


for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        b, g, r = inputImage[i, j]
       
       #non-linear rgb
        r=r/255
        g=g/255
        b=b/255
       #inv-gamma correction
        if(r<0.03928):
        	r=r/12.92
        else:
        	r=(np.power(((r+0.055)/1.055),2.4))

        if(g<0.03928):
        	g=g/12.92
        else:
        	g=(np.power(((g+0.055)/1.055),2.4))

        if(b<0.03928):
        	b=b/12.92
        else:
        	b=(np.power(((b+0.055)/1.055),2.4))


        #Convert to XYZ
        rgb=[]
        rgb.append(r)
        rgb.append(g)
        rgb.append(b)

        matrix = [[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]]
        xyz=[]
        xyz= np.dot(matrix,rgb)
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]       
        
                      
        #XYZ to xyY
        if((x+y+z)!=0):
            x2= float(x/(x+y+z))
            y2= float(y/(x+y+z))
            z2= float(y)
        else:
            x2=0
            y2=0
            z2=0
        #Linear scailing
        
        if(y2<y_min):
        	y2 = 0
        elif(y2>=y_max):
        	y2 = 1
        else:
        	y2=((y2-mn)*(1-0)/(mx-mn))

        #Convert xyY to XYZ

        try:
            x= (x2/y2)*y
            y=y
            z=((1-x2-y2)/y2)*y
        except ZeroDivisionError:
            x=0
            y=0
            z=0


        Rsrgb=(3.240479*x)+(-1.53715*y)+(-0.498535*z)
        Gsrgb=(-0.969256*x)+(1.875991*y)+(0.041556*z)
        Bsrgb=(0.055648*x)+(-0.204043*y)+(1.057311*z)

        if(Rsrgb<0):
        	Rsrgb=0
        if(Gsrgb<0):
        	Gsrgb=0
        if(Bsrgb<0):
        	Bsrgb=0
        if(Rsrgb>1):
        	Rsrgb=1
        if(Gsrgb>1):
        	Gsrgb=1
        if(Bsrgb>1):
        	Bsrgb=1

        if(Rsrgb<0.00304):
        	Rsrgb=12.92*Rsrgb
        else:
        	Rsrgb=((1.055*(np.power(Rsrgb,(1/2.4))))-0.055)

        if(Gsrgb<0.00304):
        	Gsrgb=12.92*Gsrgb
        else:
        	Gsrgb=((1.055*(np.power(Gsrgb,(1/2.4))))-0.055)
        if(Bsrgb<0.00304):
        	Bsrgb=12.92*Bsrgb
        else:
        	Bsrgb=((1.055*(np.power(Bsrgb,(1/2.4))))-0.055)

        r=Rsrgb*255
        b=Bsrgb*255
        g=Gsrgb*255
        if (math.isnan(r)==True):
            r=0
        if (math.isnan(g)==True):
            g=0
        if (math.isnan(b)==True):
            b=0            

        outputImage[i,j] = [b, g, r]

cv2.imwrite(name_output, outputImage)

