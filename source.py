# this is an implementation of the particle filter

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import time
import cv2


IMAGE_SIZE = 200
IMAGES_NUMBER = 1018
CLOUD_SIZE = 200
IMAGE_SIZE_tuple = (IMAGE_SIZE,IMAGE_SIZE)

#this function moves a given particle in a random direction
def move(X):
    x = X[0]
    y = X[1]
    n = random.randint(1,4)
    if n == 1 :
        return (x+1 , y) #right
    elif n == 2 :
        return (x , y-1) #up
    elif n==3:
        return (x , y+1) #down
    elif n == 4 :
        return (x-1 , y) #left

    
# this function cheks the color of every pixel of a given image
def mesure(img):
    z = np.zeros(IMAGE_SIZE_tuple).tolist()
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            if (img[i][j] == (255,0,0)).all():
                z[i][j] = True
            else :
                z[i][j] = False
    
    return z


# this function loads the set of images
def load():
    imgs = []
    for i in range(1,IMAGES_NUMBER):
        # set the names of the files
        if i < 10 :
            name = f'snake_000{i}.png'
        elif i < 100 :
            name = f'snake_00{i}.png'
        elif i < 1000 :
            name = f'snake_0{i}.png'
        else :
            name = f'snake_{i}.png'
        
        image = Image.open (f'./snake_color/{name}').convert('RGB')
        img_arr = np.array(image)
        #add the image to the end of the vector
        imgs.append(img_arr)
    # return the set of images
    return imgs

def animate(imgs):
        for img in imgs:
            cv2.imshow('animation', img)
            cv2.waitKey(50)
            if cv2.waitKey(1) == ord('q'):
            # press q to terminate the loop
                cv2.destroyAllWindows()
                break  
    
def gen_cloud(n_samples):
    x = np.random.randint(1 , IMAGE_SIZE , n_samples)
    y = np.random.randint(1 , IMAGE_SIZE , n_samples)
    return [(x[i],y[i]) for i in range(n_samples) ]


# particul filter function
# x is the inistial set of points , z is the mesurment matrix , n_samples is the number of samples
def particul_filter(x_init , z ):
    n_samples = len(x_init)
    x_t = x_init
    w_t = np.ones(n_samples).tolist()
    snake = []
    
    p = np.ones(n_samples).tolist()
    n_iterration  = 0
    for i in range(n_samples):
        x_t[i] = move(x_t[i])
        w_t[i] = weights(x_t[i] , z)
        if (w_t[i] == 1):
            # saving the position of the snake
            snake.append(i)
        p[i] = (w_t[i]*x_t[i][0] , w_t[i]*x_t[i][1])
    
    # resampling
    for i in range(CLOUD_SIZE/2):
        # this condition verifies that we know the position of the snake and that the particle isn't on the snake
        if (w_t[i] == 0 and len(snake)):
            # we give to this particle the position of a particle which is on the snake after moving it randomly
            a = np.random.randint(len(snake))
            p[i] = move(p[snake[a-1]])
        elif w_t[i] == 1 :
            #dont do anything - the particle remains the same
            p[i] = p[i]
        else :
            # creating a random particle
            p[i] = (np.random.randint(1,IMAGE_SIZE) , np.random.randint(1,IMAGE_SIZE))
    return p

def weights(p,z):
    pixel = list(p)
    pixel = [abs(pix) for pix in pixel]
    
    if (pixel[0] == 0 or pixel[0] >= IMAGE_SIZE) :
        pixel[0] = 1
    if (pixel[1] == 0 or pixel[1] >= IMAGE_SIZE) :
        pixel[1] = 1

    #mesurement
    if (z[pixel[0]][pixel[1]]== True):
        print(pixel)
        return 1
    else:
        return 0


imgs = load()
Z = mesure(imgs[2])
X = gen_cloud(CLOUD_SIZE)

for i in range(500) :
    # adding the particle on the image
    for x in X :
        imgs[i] = cv2.circle(imgs[i], (x[1],x[0]), radius=0, color=(0, 0, 255), thickness=-1)
    # generate the mesurment matrix for the image
    Z = mesure(imgs[i])
    # applying the particular filter
    X = particul_filter(X,Z)
    
animate(imgs)

