# CVFX HW6 Team11

Please wait a moment for your browser downloading all the gif results to play smoother.

#### Results from ORB-SLAM2
- We calibrated our camera with a simple method described in [here](https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch04.html?fbclid=IwAR0f7qhc-c8D7RrQNjl4qOsQ_xED30K9FhtgvQwdSeZTaS4v6vtbiHiGZ-Q), only considering '''cx, cy, fx, fy''' without distortion parameters.
- Because ORBSLAM2 only provides camera poses of keyframes in ```KeyFrameTrajectory.txt```,
    -  we delete those frames which are not keyframes,
    - so the below videos are composed of only keyframes instead of all frames, and thus the videos look not smooth.
- Implementation
    - in '''insert3d.py''' we use opencv function to render the 3d model without texture onto the background video
    - in '''insert3d-pygame.py''' we use pyOpenGL to render the 3d model with texture and blend the non-black region of 3D model onto the background video.
- Results

| Scene | 2d image | 3d model (no texture) | 3d model |
| :---: | :--: | :--: | :--: |
| 1 | ![](result/room-2d.gif) | ![](result/room-3d-notexture.gif) | ![](result/room-3d.gif) |




#### Results from Adobe After Effect
| Fantacy interface on the desk | A can 3D model on the table |
| :--: | :--: |
| ![](AE-result/result0.gif) | ![](AE-result/result1.gif) |


#### Comparison
Anyway, Adobe AE has easy to use UI and adding any 3D model or visual effect in AE is intuitive. On the other hand, it take tones of time for us to tune the position of the inserted objects by using only pure python.
