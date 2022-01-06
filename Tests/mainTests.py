import sys
import json

sys.path.append('../')

from src.Object_Overlay.objectOverlay import ObjectOverlay
from src.Picture_Overlay.pictureOverlay import PictureOverlay


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    
    # PictureOverlay().overlayImage(config["known_image"], config["target_image"], config["test_video"], videoOutput = False)
    # PictureOverlay().overlayVideo(config["known_image"], config["target_video"], config["test_video"], videoOutput = False)


    ObjectOverlay().render(config["known_image"], config["3d_object"], config["test_video"], config["calibration_video"], videoOutput = False)
    # ObjectOverlay().render(config["known_image"], config["3d_object_dragon"], config["test_video"], config["calibration_video"], videoOutput = False)
    # ObjectOverlay().render(config["known_image"], config["3d_object_Wood_House"], config["test_video"], config["calibration_video"], videoOutput = False)
    # ObjectOverlay().render(config["known_image"], config["3d_object_Chess_Board"], config["test_video"], config["calibration_video"], videoOutput = False)


