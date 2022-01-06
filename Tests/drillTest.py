import json
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt


def main(fname):
    fuze_trimesh = trimesh.load(fname)
    
    T = np.eye(4)
    T[0:3, 0:3] = rot_x(np.pi / 2)
    fuze_trimesh.apply_transform(T)
    
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
        [0.0, -s,   s,   0.3],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  s,   s,   0.35],
        [0.0,  0.0, 0.0, 1.0],
    ])
    
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()
    
def rot_x(t):
    ct = np.cos(t)
    st = np.sin(t)
    m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return m


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    
    main(config["3d_object"])
    # main(config["3d_object_dragon"])
    # main(config["3d_object_Wood_House"])
    # main(config["3d_object_Chess_Board"])
