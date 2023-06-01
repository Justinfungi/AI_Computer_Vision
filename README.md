# AI_ComputerVision

- Inspired by Dr Kai Han
- For recording the progress of CV learning

## Dataset:

<details>
  <summary > Expand </summary>

  ## coco
  1. https://cocodataset.org/#download
  2. https://github.com/cocodataset/cocoapi.git


  ### Data Preprocessing
  1. Tiny image strategy:

    i) https://groups.csail.mit.edu/vision/TinyImages/
    ii) https://openreview.net/pdf?id=s-e2zaAlG3I

  ```bash
  ```

</details>

# Traditional Methods

<details>
  <summary > Expand </summary>
  
  ### Stitching

      SIFT and Harris corner detection
      https://www.cs.tau.ac.il/~turkel/imagepapers/comparison_sift-harris-corner.pdf

  ```bash
  ```

</details>



# Text to 3D

<details>
  <summary > Expand </summary>

  ```bash
  ```

  ## Diffusion models

  1. DreamFusion:

      Nerf + Stable Diffusion + DMTet
      1 hour for 1 case

      https://dreamfusion3d.github.io
      They didn't provide the official code, but there is a reliable third-party reproduction you can leverage:
      https://github.com/ashawkey/stable-dreamfusion

      Personal Ammendment
      https://github.com/Justinfungi/stable-dreamfusion/tree/HKUproject

</details>



# Detection

<details>
  <summary > Expand </summary>

  ```bash
  ```

  ## Yolov7
  1. Fintuning - A detailed workflow (For starter)

  | First Header  | Second Header |
  | ------------- | ------------- |
  | Data  | Roboflow  (Suceess, easy, with UI, folder config)  |
  | Training  | !python train.py --workers 8 --device 0 --batch-size 32 --data Customization/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'Customization/yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml  |
  | Explanation | --data Customization/data.yaml: change to the folder you want, data config file. <br> |
  | Customization/data.yaml | Attach below |
  | Error1 | torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB (GPU 0; 10.76 GiB total capacity; 9.63 GiB already allocated; 36.44 MiB free; 9.75 GiB reserved in total by PyTorch) |

  ``` bash
  train: Customization/train/images #Path
  val: Customization/valid/images #Path
  nc: 4 # number of class
  names: ['Net', 'Player1', 'Player2', 'Tennis Ball'] # lable name

  ```

</details>
