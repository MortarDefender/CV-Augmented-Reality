# CV-Augmented-Reality
using open cv function we can create and map pictures and objects to the real world at real time

## System Requirments:
- python 3.6 or heighr
- cv2 library
- numpy library

## Installation:
installation can be done using conda.

```cmd
conda activate
python setup.py install
```

## Run:
Picture Overlay:

```python
from pictureOverlay import PictureOverlay
PictureOverlay().overlayImage(known_image_file_path, target_image_file_path, test_video_file_path, videoOutput)
```    

Object Overlay:
```python
from objectOverlay import ObjectOverlay
ObjectOverlay().render(known_image_file_path, 3d_object_file_path, test_video_file_path, calibration_video_file_path, videoOutput)
```


# Picture Overlay Demo:
<img src="demo/picture overlay demo.gif" height="400">

# Object Overlay Demo:
<img src="demo/3d model rendering demo.gif" height="400">

# All Models Demo
<img src="demo/models_demo/drill model.gif" height="330"> <img src="demo/models_demo/dragon model.gif" height="330"> <img src="demo/models_demo/chess board model.gif" height="330"> <img src="demo/models_demo/cube model detection.gif" height="330">
