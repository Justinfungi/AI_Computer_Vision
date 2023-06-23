# Objection Detection

Learderboard: https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new

<details>
  <summary > Expand </summary>

  ```bash
  ```
  ## Yolo Series:
  | First Header  | Second Header |
  | ------------- | ------------- |
  | Yolov1 | a single convolutional neural network (CNN) to detect objects, not accurate |
  | Yolov2 |  It used anchor boxes to improve detection accuracy and introduced the Upsample layer, which improved the resolution of the output feature map. |
  | Yolov3 | I) increasing the accuracy and speed of the algorithm. ii) Darknet-53 (stack with another 53) iii) allowing different scales and aspect ratios iv) The use of Feature Pyramid Networks (FPN) and GHM loss function |
  | Yolov4 |  a new backbone network, improvements to the training process, and increased model capacity. Cross mini-Batch Normalization, a new normalization method designed to increase the stability of the training process.  |
  | yolov5 (2020) | used the EfficientDet architecture, based on the EfficientNet network,  |
  | Yolov6 (2022) | use of a new CNN architecture called SPP-Net (Spatial Pyramid Pooling Network). This architecture is designed to handle objects of different sizes and aspect ratios, making it ideal for object detection tasks. |
  | Yolov7 (2022)| ResNeXt , multi-scale training strategy, This helps the model handle objects of different sizes and shapes more effectively;  "Focal Loss" (address the class imbalance problem) The Focal Loss function gives more weight to hard examples and reduces the influence of easy examples. |
  | Yolov8 |  |
  ## Yolov7/v8
  1. Fintuning - A detailed workflow (For starter)

  | First Header  | Second Header |
  | ------------- | ------------- |
  | Data  | Roboflow  (Suceess, easy, with UI, folder config); LabelImg Lib ; |
  | Training  | !python train.py --workers 8 --device 0 --batch-size 32 --data Customization/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'Customization/yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml  |
  | Explanation | --data Customization/data.yaml: change to the folder you want, data config file. <br> |
  | Customization/data.yaml | Attach below |
  | Error1 | torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB (GPU 0; 10.76 GiB total capacity; 9.63 GiB already allocated; 36.44 MiB free; 9.75 GiB reserved in total by PyTorch) <br><br> Reduce the `batch_size`, Lower the Precision, Do what the error says, Clear cache, Modify the Model/Training |
  | Error2 | Envs <br><br> Dont support torch>=1.7.0,!=1.12.0 torchvision>=0.8.1,!=0.13.0 |
  | Result | Optimizer stripped from runs/train/yolov7-custom14/weights/last.pt, 74.8MB <br><br> python detect.py --weights runs/train/yolov7-cus4/weights/best.pt --conf 0.25 --img-size 640 --source Test3.png |
  | Issue 1 Similar people are having same ID | Solution by chatgpt / Tracker: https://github.com/JackWoo0831/Yolov7-tracker |
  | Issue 2 | Deepsort fail to id the object consistently in different scene |

  ``` bash
  train: Customization/train/images #Path
  val: Customization/valid/images #Path
  nc: 4 # number of class
  names: ['Net', 'Player1', 'Player2', 'Tennis Ball'] # lable name

  ```
  ![image](https://github.com/Justinfungi/AI_Computer_Vision/assets/79019929/8f5ff4fb-7417-46b5-b647-1ea9c207cb7f)
  ![image](https://github.com/Justinfungi/AI_Computer_Vision/assets/79019929/852698a0-3f33-4af7-b43c-eb934c080a00)


</details>
