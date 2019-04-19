#Python file for working functions.

#Necessary modules
from scipy import spatial
from scipy import ndimage
from scipy import stats
from math import *
import numpy as np
import copy
import os
import cv2
import operator
import numpy.ma as ma
import time
import pickle as pkl

##Classes
class fly_data(object):
    """fly_data gathers the position and orientation of a given fly all along the video."""
    def __init__(self, data_dict, identity):
        self.length = data_dict['orientations'][0]
        self.positions = data_dict['trajectories'][:,identity] #The position of the fly for each frame in pixels of the video.
        self.identity = identity
        self.directions = data_dict['directions'][:,identity]
        self.orientations = data_dict['orientations'][:,identity]
        
    def __str__(self):
        print(self, 'is a fly_data object')
        print('Number of frames : ', self.length)
        print('Identity in the video : ',self.identity)

        
    def ori_correc(self, overwrite=False): #Outputs corrected orientations based on inputted directions.
        #Should be given two arrays of numbers

        correc = convert180to360(self.orientations) #Put orientations in a 360 degrees format
        flipped = flip(correc) #Flips orientations
        minus = flip(correc, -90)
        plus = flip(correc, 90)

        #Calculate correlation for the 4 possibilities
        correl = dict()
        correl['flipped'] = ma.corrcoef(ma.masked_invalid(flipped), ma.masked_invalid(self.directions))[1,0]
        correl['minus'] = ma.corrcoef(ma.masked_invalid(minus), ma.masked_invalid(self.directions))[1,0]
        correl['plus'] = ma.corrcoef(ma.masked_invalid(plus), ma.masked_invalid(self.directions))[1,0]
        correl['correc'] = ma.corrcoef(ma.masked_invalid(correc), ma.masked_invalid(self.directions))[1,0]

        #get optimum
        opti = max(correl.items(), key=operator.itemgetter(1))[0]

        #Outputs better corrected orientations
        if opti == 'flipped':
            print("Orientations have been corrected and flipped 180 degrees.")
            print("Correlation is ", correl['flipped'])
            if overwrite == True:
                self.orientations = flipped
            else:
                return flipped

        elif opti == 'minus':
            print("orientations have been corrected and flipped -90 degrees")
            print("Correlation is ", correl['minus'])
            if overwrite == True:
                self.orientations = minus
            else:
                return minus

        elif opti == 'plus':
            print("orientations have been corrected and flipped +90 degrees")
            print("Correlation is ", correl['plus'])
            if overwrite == True:
                self.orientations = plus
            else:
                return plus

        elif opti == 'correc':
            print("orientations have been corrected only")
            print("Correlation is ", correl['correc'])
            if overwrite == True:
                self.orientations = correc
            else:
                return correc
            
            
class relative_fly(object): #Should be given two fly objects, and calculate relative distance, speed, angle from focus to other fly.
    
    def __init__(self, focusfly, newfly):
        """Should be given two fly_data objects. Produces relative position and angle of the second fly to the first."""
        self.positions = newfly.positions - focusfly.positions
        xs = self.positions[:,0]
        ys = self.positions[:,1]
        
        self.dist, absol_angle = angleFromPoints(xs,ys)
        angle = absol_angle - focusfly.orientations
        angle[angle < 0] += 360
        self.angle = angle
        self.speed = self.dist[1:] - self.dist[:-1]
        self.speed = np.insert(self.speed, 0,0)

        
class relative_set(object): #Givne a list of fly_data objects, calculates and outputs the relative_fly of every fly to the others.
    def __init__(self, flystack):
        pos_stack = np.empty([1,2])
        dist_stack = np.empty([])
        angle_stack = np.empty([])
        
        for focus in flystack:
            for fly in flystack:
                if focus != fly:
                    rel = relative_fly(focus, fly)
                    angle = np.radians(rel.angle)
                    ys = rel.dist*np.sin(angle)
                    xs = rel.dist*np.cos(angle)
                    pos = np.column_stack([xs,ys])
                    pos_stack = np.vstack([pos_stack, pos])
                    dist_stack = np.hstack([dist_stack, rel.dist])
                    angle_stack = np.hstack([angle_stack, rel.angle])
        pos_stack = pos_stack[1:,]            
        self.positions = pos_stack
        self.dist = dist_stack
        self.orientations = angle_stack


#Functions

def get_ori(Dir, video, traj, thresh = 50, noise_thresh=0.5): #Gets raw orientations and directions from numpy array of positions and video file.
    #Both .npy file and video file should be in the same Dir folder.
    
    #Fantastic files and where to find them
    path_to_vid = r'{}{}{}'.format(Dir, os.sep, video)
    path_to_traj = r'{}{}{}'.format(Dir, os.sep, traj)
    cap = cv2.VideoCapture(path_to_vid)

    trajectories_dict = np.load(path_to_traj, encoding='latin1').item() #Trajectorie data from idtrackerai.
    trajectories = trajectories_dict['trajectories']
    
    #Some information that will be useful later
    nberflies = len(trajectories_dict['trajectories'][0]) #Get number of individuals from trajectories
    framerate = cap.get(cv2.CAP_PROP_FPS)
    
    i = 0 #Index for taking the right trajectories.

    while(cap.isOpened()):
    #for _ in range(500): #To be used to make it run faster if needed
        
        #Take the next image from the video and turn it grey
        ret, img = cap.read()
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        else: #If no more images, break the loop
            cap.release()
            continue

    #Set the threshold and get the binary image        
        ret, diff_img = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)

                #Get a bunch of contours from the binary
        contours,hierarchy = cv2.findContours(diff_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


                #For every contours, get its center, and put contour in a dict with center as key
        centers = list()
        right_contours = list()

        for c in contours:
            if len(c)>15 and len(c)<100: #contour should be in a reasonable size range
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY]) #Add center to the list
                right_contours.append(c) #Add corresponding contour to a similar list

                #Get the trajectory data from idtrackerai
        positions = trajectories[i]

        ori_frame = list() #A list of orientations for the current frame

        for fly in positions:
            if (np.isfinite(fly[0])):
                ind = spatial.KDTree(centers).query(fly)[1]
                ellipse = cv2.fitEllipse(right_contours[ind]) #Fitting the ellipse
                orientation = np.float(ellipse[2])
                ori_frame.append(orientation)
                
            else:
                ori_frame.append(np.nan)
        ori_frame = np.array(ori_frame)
        try:
            ori_stack
        except NameError:
            ori_stack = ori_frame
        else:
            ori_stack = np.vstack([ori_stack, ori_frame])
 
    #Get directions part :
        shape_pos = np.arange(nberflies)
        if i == 0: #If first iteration, everything is zero
            direc_stack = np.zeros_like(shape_pos)
            speed_stack = np.zeros_like(shape_pos)
            dist_stack = np.zeros_like(shape_pos)
            direc = np.zeros_like(shape_pos)
        elif isinstance(i, int): #If not the first iteration
            dist, direc = get_angle(trajectories, i, direc, noise_thresh)
            speed = dist*framerate

            dist_stack = np.vstack([dist_stack, dist]) # distance traveled in pixels            
            speed_stack = np.vstack([speed_stack, speed]) #Speed in pixels per second
            direc_stack = np.vstack([direc_stack, direc]) #direction in degrees
        
        
        i+=1 #An index so that we take the right trajectories for every frame.
        
    data = dict() #Dictionary in which we will put the three stacks of data.
    data['trajectories'] = trajectories
    data['speeds'] = speed_stack
    data['directions'] = direc_stack
    data['orientations'] = np.array(ori_stack)
    data['mov'] = dist_stack
    data['contours'] = right_contours
    
    flystack = list()
    for fly in range(nberflies):
        
        flystack.append(fly_data(data, fly))
    
    return flystack
        
def get_angle(trajectories, frameID, previous_angles, noise_thresh=0.5):
                #Outputs direction, speed and distance for a given frame.
    
    positions = trajectories[frameID]

    oldpositions = trajectories[frameID-1] #Get previous values
    delta = positions - oldpositions #Compute difference between two iterations
    xs = delta[:,0]
    ys = delta[:,1]
    #Calculate angle and distance
    dist, angle = angleFromPoints(xs, ys)
    for i in range(len(delta)):
        x = xs[i]
        y = ys[i]
        #Remove noise
        if (x > -noise_thresh) & (x < noise_thresh) &\
        (y > -noise_thresh) & (y < noise_thresh):
            dist[i] = 0
            angle[i] = previous_angles[i]
            
    return(dist, angle)



def convert180to360(ori): #Should be given an array of numbers
    
    ori_conv = copy.deepcopy(ori)
    diffs = ori_conv[1:] - ori_conv[0:-1]
    bool = True
    for i in range(len(diffs)):
        if abs(diffs[i]) > 90 and bool == True:
            ori_conv[i+1:] += 180
            bool = False
        elif abs(diffs[i]) > 90 and bool == False:
            ori_conv[i+1:] -= 180
            bool = True
    return(ori_conv)

def flip(ori, angle=180): #Should be given an array of numbers and a single value
    
    ori_flip = copy.deepcopy(ori)
    ori_flip += angle
    for i in range(len(ori_flip)):
        if ori_flip[i] >= 360:
            ori_flip[i:] -= 360
        elif ori_flip[i] <= 0:
            ori_flip[i:] +=360
    return(ori_flip)


def direc_stats(data, chunkrange, fly):     #Get stats about a chunk
                                            #Not really useful atm

    tronc_direc = data['directions'][chunkrange,fly]
    ori = data['orientations'][chunkrange,fly]
    print("Stats for fly {} from frame {} to {}".format(fly, chunkrange[0], chunkrange[-1]))
    print("Variance of the direction over the chunk is :", np.var(tronc_direc))
    print("Cross correlations : ",np.correlate(tronc_direc, ori))
    print("Correlation coef : ",np.corrcoef(np.vstack([tronc_direc, ori]))[1,0], '\n')
    plt.figure(figsize=(20,5))
    plt.ylim([-10,370])
    plt.plot(ori)
    plt.plot(tronc_direc)
    
def angleFromPoints(x,y): #Gets two arrays of x/y distances between two points
                          #Can do case by case or batch treatment.  
    
    dist = np.sqrt((x*x)+(y*y)) #Distance between each couple of points
    angle = np.degrees(np.arctan(np.absolute(y)/np.absolute(x)))
    #Angle correction based on negativity of coordinates.
    #This ensure angle range on 0:360 and correspond to trigonometric circle. 0 is looking on the right.
    for i in range(len(angle)):
        if   x[i] <0 and y[i] <0:
            angle[i] += 180 
        elif x[i] <0:
            angle[i] = 180 - angle[i]
        elif y[i] <0:
            angle[i] =360 - angle[i]
    return dist, angle


#Handling the data in a "packaged" way

def save_ori_pickle(session_name):
    Dir = "/home/maubry/python/idtrackerai/raw"
    st = time.time()
    video = session_name+".avi"
    traj = session_name+".npy"
    pkl_name = session_name+"_ori_data.pkl"

    flystack = get_ori(Dir, video, traj)
    f = open(pkl_name, "wb")
    pkl.dump(flystack, f)
    laps = time.time() - st
    print('time elapsed : {}min and {}sec'.format(laps//60, int(laps%60)))

    
def get_ori_pickle(session_name):
    pkl_name = session_name+"_ori_data.pkl"
    f = open(pkl_name, "rb")
    random_stack = pkl.load(f)
    return random_stack
    
    
    
    