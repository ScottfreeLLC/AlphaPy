# Table-tennis-ball-3D-trajecture-estimation
This repo makes use of twin regular network camera for 3D table tennis trajectory calculation.  We first calibrate the twin cameras using a DLT(Direct Linear Transformation) method for ball’s 3D coordinates derivation.  Then, we applied a vision algorithms to grab the 2D ball centers synchronously.  Finally, a 3D trajectory is synthesized and visualized.  The following introduces the DLT calculation process.
DLT is the abbreviation of direct linear transformation, which is mainly used for stereo correction of multiple cameras. The relationship is shown in (1). When the corresponding point exceeds 6, it is written in matrix form as in formula (2), u1, v1 to uN, vN represent N corresponding image coordinates, and the corresponding 3-dimensional coordinates are x1, y1, z1 to xN ,yN,zN. The L value is the L value of formula (1), N>6.
 (1)
 
 ![image](https://user-images.githubusercontent.com/33441535/168031943-3ed5e03b-7af9-4758-a319-4275583c898f.png)

 (2)
 
 ![image](https://user-images.githubusercontent.com/33441535/168031960-031c6efb-4f9d-4bb7-b073-ea73724b5843.png)

Equation (2) can be simplified by Equation (3), and L is solved by the least square difference of Equation (3), where U represents the left side of the equal sign, and L represents the matrix of L1~L11.
U=AL                                 (3)
Equation (3) can be simplified by Equation (4), and L is solved by the least square difference of Equation (4), where U represents the left side of the equal sign, and L represents the matrix of L1~L11.


![image](https://user-images.githubusercontent.com/33441535/168032004-46d093b1-4e65-4a74-84cc-32817f34f540.png)
          (4)
Once there are multiple cameras (at least two) and the corresponding L matrix is solved, the corresponding estimation (x, y, z) can be represented by equation (5), which is shown in equation (6) after simplification. At this time, u, v represent the coordinates of the object captured by multiple cameras, such as the center of the ball, etc. The different u, v values in individual cameras are brought into the L value of different cameras. Formula (5) is the same as the prediction. The relationship between 3-dimensional coordinates. Equation (7) uses the least squares sum to calculate the predicted (x, y, z) coordinates. If more cameras are used in this type, the accuracy will be improved.


![image](https://user-images.githubusercontent.com/33441535/168032044-41246ed0-9dab-4bf1-a4dc-d714fa73603b.png)

(5)Y=AX                   

(6)


![image](https://user-images.githubusercontent.com/33441535/168032061-5129ee11-f417-492f-a607-e93ac7849813.png)

(7)

This experiment first uses Hough map detection to convert the dual-camera color channel to the Hough channel, so that the position of the sphere can still be grasped under high-speed motion. Then use the DLT linear transformation correction to calculate the ball’s 3D coordinates.
Figure 1 shows the position of the TT ball identified by color, which is suitable for static ball coordinate identification. Figure 2 shows the position of the TT ball identified by Hough diagram, which is suitable for dynamic ball coordinate identification. The camera can still capture ball position under high-speed motion. The camera specification used for this study is Logitech StreamCamFull HD 1080P/60FPS.
![image](https://user-images.githubusercontent.com/33441535/168032110-b0ab4d5a-0c76-4057-a6cd-14ed7980956d.png)
 
Fig. 1. Pool positions identified by color (static)

![image](https://user-images.githubusercontent.com/33441535/168032131-75cf2adb-a882-480f-836a-868928e3d698.png)
  
Fig. 2. The position of the billiards identified by the Hough diagram (dynamic)

![image](https://user-images.githubusercontent.com/33441535/168032171-cfc3986b-085a-402e-8ba2-07114425653d.png)

![image](https://user-images.githubusercontent.com/33441535/168032185-0d9c9744-0c50-453a-9d65-23fe79731621.png)
    
Fig. 3 dual-camera stereo DLT calibration result.
Figure 3 shows the dual-camera stereo DLT calibration setting environment, including a calibration board and dual cameras. There are about 2~300 known 3D coordinate points on the calibration board. The calibrated 3D coordinates of the checker board after DLT calculation are shown on the right in Figure 3.

![image](https://user-images.githubusercontent.com/33441535/168032247-69c84787-345c-4dd8-99cb-b3421e45437b.png)

![image](https://user-images.githubusercontent.com/33441535/168032261-9b9165e8-251d-4c5d-8bd9-c8c116a18d21.png)
  
Fig. 4

![image](https://user-images.githubusercontent.com/33441535/168032282-21a32c80-047a-4179-8cc9-cb1e83be6ed4.png)

![image](https://user-images.githubusercontent.com/33441535/168032300-752a99cd-4e13-433f-b1cd-f7ef9d1a4e74.png)
   
Fig. 5
Figure 4 shows the tracking result of the ball trajectory of the twin camera, showing two motion trajectories, using the same launch angle and speed, the trajectories are roughly the same when viewed from two angles, but the difference is about 10mm, but it is about the same when viewed from a vertical angle. The plane shows that the correction result is correct, and the rebound point can also be seen. Figure 5 shows the different launch angles and velocities. It can be seen that their trajectories are completely different, but they are about the same plane when viewed from a vertical angle, which is consistent with the actual situation.  The purpose of using the automatic ball firing machine in this stage is to verify the correctness of the DLT program. The ball machine can be used to repeat the path characteristics to verify the results of the DLT calculation.

The committed code explains the process of capturing the ball image and derive the 3D trajectory of a table tennis ball.  The driver code is listed below for explanation:
imgfindcolor(cap0);#cam1 capture the ball center1
imgfindcolor1(cap1);#cam2 capture the ball center2

while True:
    # Output parameters
    img0 = imgfindcolor(cap0) #cam1 capture the ball center1
    img1 = imgfindcolor1(cap1) #cam2 capture the ball center2
    cv2.imshow('cap1', img0) # Screen display of cam1
    cv2.imshow('cap2', img1) # Screen display of cam2


#Calculate ball position coordinates 

    A = np.array([                          
            [ 0.4855,   0.8075,  -0.2054], 
            [-0.2805,   0.0139,  -0.8023],
            [-1.3411,  -0.7169,   0.1418],
            [ 0.1980,  -0.0016,  -0.6909]     
])#the array A from eq. 2+3

    Y = np.array([ targetPos_x - 168.6745, targetPos_x - 851.4135, targetPos1_x - 1716.9, targetPos1_y - 704.5 ]) #the array Y from 5+6
    X = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(Y))#least square error eq. 7

    print(X)   # given the pairs of ball center coordinates from 2 cameras use DLT to derive the 3D coordinates of ball
