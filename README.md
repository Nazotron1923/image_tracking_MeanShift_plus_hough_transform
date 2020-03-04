# image_tracking_MeanShift_plus_hough
---


Thus, in this work, two algorithms (**Mean Shift** and **Hough transform**)
were examined for video analysis and object tracking. During the
experiences, advantages and disadvantages of both approaches have been identified. 
The main advantage of Mean Shift is its speed of operation. The
main advantage of Hough Transform is its precision and ability to
take into account the deformation of objects. Not the last role has the quality of
the video (number of frames per second). Through the use of the library
numpy we managed to speed up Hough's transformed algorithm. Possibility
combining the two approaches was discussed.


See the report

White rectangle - Hough transform, blue - MeanShift.

|         |            |   |
| ------------- |:-------------:| -----:|
|![Video_VOT-Car1](/img_res/h_Video_VOT-Car_Frame_150.png)|![Video_VOT-Car2](/img_res/h_Video_VOT-Car_Frame_200.png) | ![Video_VOT-Car3](/img_res/h_Video_VOT-Car_Frame_250.png) |
|![Video_Antoine_Mug1](/img_res/h_Video_Antoine_Mug_Frame_1.png)|![Video_Antoine_Mug2](/img_res/h_Video_Antoine_Mug_Frame_25.png) | ![Video_Antoine_Mug3](/img_res/h_Video_Antoine_Mug_Frame_50.png) |
|![Video_VOT-Woman1](/img_res/h_Video_VOT-Woman_Frame_1.png)|![Video_VOT-Woman2](/img_res/h_Video_VOT-Woman_Frame_75.png) | ![Video_VOT-Woman3](/img_res/h_Video_VOT-Woman_Frame_150.png) |
