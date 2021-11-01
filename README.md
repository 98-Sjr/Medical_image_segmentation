# github_project
 
  This is a project for medical image segmentation. This project includes common medical image segmentation models such as U-net, FCN, Deeplab, SegNet, PSPNet and so on.  
The use steps are as follows:  
    (1): Place dataset in 'inputs' folder.  
    (2): Modify the path in 'preprocess.py' and run it to generate the image with uniform size.  
    (3): Modify the parameter in 'train.py' and run it to obtain the trained model.  
    (4): Place the image you want to segment in 'test' folder.   
    (5): Modify the parameter in 'val.py' and run it to obtain the predicted image, and it will saved in 'test' folder.
