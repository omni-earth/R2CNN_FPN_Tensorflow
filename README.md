# R2CNN: Rotational Region CNN for Orientation Robust Scene Detection

A Tensorflow implementation of FPN or R2CNN detection framework based on [FPN](https://github.com/yangxue0827/FPN_Tensorflow).  
You can refer to the papers [R2CNN Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/abs/1706.09579) or [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)    
Other rotation detection method reference [R-DFPN](https://github.com/yangxue0827/R-DFPN_FPN_Tensorflow), [RRPN](https://github.com/yangJirui/RRPN_FPN_Tensorflow) and [R2CNN_HEAD](https://github.com/yangxue0827/R2CNN_HEAD_FPN_Tensorflow)         

# Citing [R-DFPN](http://www.mdpi.com/2072-4292/10/1/132)

If you find R-DFPN useful in your research, please consider citing:

    @article{yangxue_r-dfpn:http://www.mdpi.com/2072-4292/10/1/132
        Author = {Xue Yang, Hao Sun, Kun Fu, Jirui Yang, Xian Sun, Menglong Yan and Zhi Guo},
        Title = {{R-DFPN}: Automatic Ship Detection in Remote Sensing Images from Google Earth of Complex Scenes Based on Multiscale Rotation Dense Feature Pyramid Networks},
        Journal = {Published in remote sensing},
        Year = {2018}
    }  

# Configuration Environment
ubuntu(Encoding problems may occur on windows) + python2 + tensorflow1.2 + cv2 + cuda8.0 + GeForce GTX 1080     
If you want to use cpu, you need to modify the parameters of NMS and IOU functions use_gpu = False  in cfgs.py     
You can also use docker environment, command: docker pull yangxue2docker/tensorflow3_gpu_cv2_sshd:v1.0    

# Installation      
  Clone the repository    
  ```Shell    
  git clone https://github.com/omni-earth/R2CNN_FPN_Tensorflow.git
  ```     
  ```Shell
  $R2CNN_ROOT is equivalent to your current working directory where this repo was pulled to,
  e.g. /mnt/cirrus/models/R2CNN_FPN_Tensorflow/
   ```
  Make sure this full path is defined in libs/configs/cfgs.py
  

# Make tfrecord     
The data is VOC format, reference [here](sample.xml)     
data path format  ($R2CNN_ROOT/data/io/divide_data.py)    
belmont_rotated  
>belmont_rotated  
>>Annotation  
>>JPEGImages   

>belmont_rotated   
>>Annotation   
>>JPEGImages   

Clone the repository    
  ```Shell    
  cd $R2CNN_ROOT/data/io/belmont_rotated/  
  python convert_data_to_tfrecord.py --VOC_dir='./' --save_name='train' --img_format='.jpg' --dataset='building'
       
  ``` 
 OR just pull this tfrecords example for belmont rotated buildings to $R2CNN_ROOT/data/tfrecords/ 
 ```Shell
 aws s3 cp s3://oe-evp/R2CNN_FPN_Tensorflow/tfrecords/building_top300_051419_successfull10kIter.tfrecord $R2CNN_ROOT/data/tfrecords/ 
 ```
 
# Train   
1. (If needed) Modify $R2CNN_ROOT/libs/lable_name_dict/***_dict.py, corresponding to the number of categories in the configuration file    
2. Download pretrained weights ([resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) or [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)) from [here](https://github.com/yangxue0827/models/tree/master/slim), then extract to folder $R2CNN_ROOT/data/pretrained_weights    
3.  
  ```Shell    
  cd $R2CNN_ROOT/tools      
  ``` 
4. Run R2CNN)     

   ```Shell    
  python train1.py   
  ``` 

# Infer   
1. Trained weights are stored in $R2CNN_ROOT/output/res101_trained_weights/v5/   
2. put images for inference in $R2CNN_ROOT/tools/inference_image/     
     
  ```Shell    
  cd $R2CNN_ROOT/tools      
  ```    

If you want to run inference with R2CNN:   
  
  ```Shell    
  python inference1.py   
  ```    

# Test tfrecord     
  ```Shell    
  cd $R2CNN_ROOT/tools   
  python test.py(test1.py)   
  ```    

# eval   
  ```Shell    
  cd $R2CNN_ROOT/tools   
  python eval.py(eval1.py)  
  ```  

# Summary    
  ```Shell    
  tensorboard --logdir=$R2CNN_ROOT/output/res101_summary/ 
  ```    

# Graph
![04](graph.png) 
    
