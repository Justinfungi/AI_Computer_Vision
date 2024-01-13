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

  ### Detection:

      Line Detection:

      Hough Transform (HT)
      APAI3010

      Progressive Probabilistic Hough Transform (PPHT):
      It is an improvement over the traditional Hough Transform and works faster because it examines a randomly chosen subset of points with every iteration.

      Randomized Hough Transform (RHT):
      This algorithm randomly selects points from an image and constructs line segments, and therefore can reduce computation time while still maintaining accuracy.

      Radon Transform:
      This transform is designed to detect straight lines within an image and can be used on binary images to increase the robustness of Hough transform. It transforms an image into a parameter space where the presence of a line is easier to detect.

      Line Segment Detector (LSD):
      It is an edge-based line detection algorithm that can detect multiple straight lines within an image in real-time.

</details>

# Text to img
<details>
  <summary > Expand </summary>
  1. Stable Diffusion
  2. Lora / Dreambooth:
     low rank finetuning methods
     
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
