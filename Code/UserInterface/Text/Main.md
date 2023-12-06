# Heart rate estimation in video data
This framework is part of a testing enviroment for *Remote Heart-Rate* estimations. 
It includes the following methods for heart rate estimations
- Eulerian Magnification
- Custom Eulerian Magnification
- HR-CNN runner
- MTTS-CAN runner

The framework allows the user to both use pre-recorded video, and live feed from the webcam using a *WEBRTC Server*.

The code and project was developed by: 

Frederik Peetz-Schou Larsen and Gustav Gamst Larsen as part of *Course 02830 - Advanced Project in Digital Media Engineering*


## GitHub Repository 
All the source code including development branches can be found through our GitHub Repository. A link can be found [here](https://github.com/s180820/Eulerian_Magnificaiton):  


## Abut the application
The application is split into three parts. 
A demo model with a predefined video, ground truth data and pre-run deep learnig data. A live feed tab allowing you to visiualise the traditional Eulerian Magnification approach aswell as the custom implementation with shared buffers allowing multiple persons to be present. Finally a pre-recorded video setting where you can upload a video and run either of the four implementations. 


### Disclaimer

Live versions of Deep learning frameworks are not available. In addition we've restricted the upload size as the methods are computationally heavy. We don't assume GPU based hardware, so the implementations are run on the CPU. 

Furthermore the statistics run are run for 200 frames due to memory constraints.

