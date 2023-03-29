#!/usr/bin/env python
# coding: utf-8

# In[146]:


#Kalman Filter
import numpy as np

class KalmanFilter(object):
    """This class implements the Kalman Filter keeps, that keeps
    track of the estimated state of the system and the uncertainties in the estimate."""

    def __init__(self):
        """Constructor to initialize the variables used by the Kalman Filter class"""
        self.dt = 0.01 # delta time
        self.X = np.array([[0], [0], [0], [0], [2], [2], [2], [2]])
        self.b = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, 0, 0, self.dt, 0, 0, 0], [0, 1, 0, 0, 0, self.dt, 0, 0], [0, 0, 1, 0, 0, 0, self.dt, 0], [0, 0, 0, 1, 0, 0, 0, self.dt], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])
        self.P = np.diag((0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01))
        self.Q = np.eye(self.X.shape[0])
        self.R = np.eye(self.b.shape[0])

        '''X: State estimate at previous step
           b: Input effect matrix
           A: State transition matrix
           H: Measurement matrix
           P: State covariance matrix
           Q: Process noise covariance matrix
           R: Observation noise matrix
        '''

    def predict(self):
        # Predict state vector X and state covariance matrix, P
        # Predicted state estimate
        self.X = np.round(np.dot(self.A, self.X))
        # Predicted estimate covariance
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        self.lastResult = self.X  # same last predicted result
        return self.X

    def correct(self, b, flag):
        #Correct or update state vector X and state covariance matrix, P
        
        if not flag:  # update using prediction
            self.b = np.dot(self.H ,self.lastResult)
        else:  # update using detection
            self.b = b

        C = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        if np.linalg.det(C):
            self.K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(C)))

        self.X = np.round(self.X + np.dot(self.K, (self.b - np.dot(self.H, self.X))))
        
        self.P = np.dot((1-np.dot(self.K, self.H)), self.P)
        self.lastResult = self.X
        return self.X


# In[147]:


# Tracker using Kalman Filter and the Hungarian Algorithm 
import numpy as np
from common import dprint
from scipy.optimize import linear_sum_assignment


class Track(object):
    #Track class for object tracking
    def __init__(self, prediction, trackIdCount):
        """Constructor to initialize variables used by the track class that takes inputs as
           the predicted centroids of the object to be tracked and identification of each track object (TrackIDCount)
        """
        self.track_id = trackIdCount              # identification of each track object
        self.KF = KalmanFilter()                  # KF instance to track this object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0                   # number of frames skipped undetected
        self.trace = []                           # trace path


class Tracker(object):
    #Tracker class to update the track vectors of the objects that are tracked
    
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,trackIdCount):
        """Constructor to initialize variables used by Tracker class, that takes the following arguments:
            1. dist_thresh: Distance threshold. Track will be deleted and new track is created when distance exceeds threshold. 
            2. max_frames_to_skip: maximum frames that can be skipped for undetected track objects
            3. max_trace_length: length of the trace path history 
            4. trackIdCount: identification of each track object
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):
        """Update the vector of tracks using following steps:
            - If no tracks vector is found, create tracks 
            - Calculate cost function
            - Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
            1. Identify tracks with no assignment (if any)
             - If tracks are not detected for long time, remove them
            2. Now look for unassigned detects
            3. Start new tracks
            4. Update KalmanFilter state, lastResult and tracks trace
        Argument:
            detections: detected centroids of the object to be tracked
        """

        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost function
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                        diff = self.tracks[i].prediction[:4] - detections[j]
                        distance = np.sqrt(diff[0][0]*diff[0][0] + diff[1][0]*diff[1][0])
                        cost[i][j] = distance
                        pass
            dist_max = max(cost[i])
            if dist_max:
                #print(cost[i])
                for j in range(len(detections)):
                    try:
                        cost[i][j] = (cost[i][j])/dist_max
                    except:
                        pass
            else:
                pass
        #print(cost)
        # Averaging the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurement to predicted tracks
        assignment = []
        if not np.isnan(cost[0][0]):
            for _ in range(N):
                assignment.append(-1)
                row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(len(row_ind)):
                assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh) and (cost[i][assignment[i]] !=1):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    dprint("ERROR: id is greater than length of tracks")

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

        # Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]], 1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(
                                            np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction


# In[150]:


#Detect Objects in a video frame
#Import python libraries
import numpy as np
from cv2 import cv2

# set to 1 for pipeline images
debug = 1


class Detectors(object):
    
    #Detectors class to detect objects in video frame
    
    def __init__(self):
     #Constructor to initialize variables used by Detectors class
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        """
        kernel = np.ones((4,4),np.uint8)

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (debug == 0):
            cv2.imshow('gray', gray)
        
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        fgmask = self.fgbg.apply(blur)
        if (debug == 1):
            cv2.imshow('bgsub', fgmask)
            
        # Retain only edges within the threshold
        _, thresh = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     
        if (debug == 0):
            cv2.imshow('thresh', thresh)

        centers = []
        # vector of object centroids in a frame
        # Find centroid for each valid contours
        for contour in contours:
      
            try:
                (x,y,w,h)=cv2.boundingRect(contour)
                if (cv2.contourArea(contour)>1500): #and cv2.contourArea(contour)<5000):
                    image= cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    b = np.array([[(x+w/2)], [(y+h/2)], [w/2], [h/2]])
                    centers.append(np.round(b))
                     
            except ZeroDivisionError:
                pass

        return centers


# In[151]:


# Object Tracking
# Import python libraries
import cv2
import copy
import time
import numpy as np


def main():
    # Main function for multi object tracking
    # Create opencv video capture object
    cap = cv2.VideoCapture('dice.webm')
    #cap = cv2.VideoCapture('car-overhead-3.avi')
    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    tracker = Tracker(0.7, 7, 30, 1)

    # Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False
    
    # Infinite loop to process video frames

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
     
        # Make copy of original frame
        orig_frame = copy.copy(frame)
        x1=0
        y1=0
        
     
        # Detect and return centeroids of the objects in the frame
        centers = detector.Detect(frame)

        # If centroids are detected then track them
        if (len(centers) > 0):

            # Track object using Kalman Filter
            tracker.Update(centers)

            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for i in range(len(tracker.tracks)):
                s=str(tracker.tracks[i].track_id)
                clr = tracker.tracks[i].track_id % 9
                      
        
                cv2.putText(frame,s,(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
                            
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j+1][0][0]
                        y2 = tracker.tracks[i].trace[j+1][1][0]
                        #print([x1, y1, x2, y2])
                        
                        
                        if not np.isnan(x2):
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),track_colors[clr], 2)
                            
                           
        # Display the resulting tracking frame
        cv2.imshow('Tracking', frame)
        # Display the original frame
        # cv2.imshow('Original', orig_frame)

        # Check for key strokes
        k = cv2.waitKey(40) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break
    # When everything is done, release the capture

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()


# In[ ]:





# In[ ]:




