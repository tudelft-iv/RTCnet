# RTCnet
RTCnet is a radar based, single-frame, multi-class detection method for moving road users (pedestrian, cyclist, car), which utilizes low-level radar cube data.
The method {provides} class information both on the {radar target- and object-level}. {Radar targets are classified} individually 
after extending the target features with a cropped block of the 3D radar cube around their positions, 
thereby capturing the motion of moving parts in the local velocity distribution.
A Convolutional Neural Network (CNN) is proposed for this classification step.
Afterwards, object proposals are {generated} with a class-{specific} clustering step, which not only considers the radar targets' positions and velocities, but their calculated class scores as well.

# This repository
 

# Cite
To cite this code or the method, please use:
