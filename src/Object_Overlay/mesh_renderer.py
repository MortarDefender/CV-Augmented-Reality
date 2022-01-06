import cv2
import trimesh
import pyrender
import numpy as np


class MeshRenderer:
    def __init__(self, K, video_width, video_height, obj_path, standing = True):
        self.K = K
        self.video_width = video_width
        self.video_height = video_height
        
        # self.initObject(obj_path)

        mesh = trimesh.load(obj_path)
        # normalize bounding box from (0,0,0) to max(30)
        mesh.rezero()  # set th LOWER LEFT (?) as (0,0,0)
        T = np.eye(4)
        T[0:3, 0:3] = 10 * np.eye(3) * (1 / np.max(mesh.bounds))
        mesh.apply_transform(T)
        
        if standing:
            # rotate to make the drill standup
            T = np.eye(4)
            T[0:3, 0:3] = self.rot_x(np.pi / 2)
            mesh.apply_transform(T)

        # rotate 180 around x because the Z dir of the reference grid is down
        T = np.eye(4)
        T[0:3, 0:3] = self.rot_x(np.pi)
        mesh.apply_transform(T)
        # Load the trimesh and put it in a scene
        mesh = pyrender.Mesh.from_trimesh(mesh) 
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        scene.add(mesh)

        # add temp cam
        self.camera = pyrender.IntrinsicsCamera(
            self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], zfar=10000, name="cam"
        )
        light_pose = np.array(
            [
                [1.0, 0, 0, 0.0],
                [0, 1.0, 0.0, 10.0],
                [0.0, 0, 1, 100.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        
        self.cam_node = scene.add(self.camera, pose=light_pose)

        # Set up the light -- a single spot light in z+
        light = pyrender.SpotLight(color=255 * np.ones(3), intensity=3000.0 * 5, innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=light_pose)

        self.scene = scene
        self.r = pyrender.OffscreenRenderer(self.video_width, self.video_height)
        # add the A flag for the masking
        self.flag = pyrender.constants.RenderFlags.RGBA
    
    def initObject(self, obj_path):
        fuze_trimesh = trimesh.load(obj_path)
        
        fuze_trimesh.rezero()  # set th LOWER LEFT (?) as (0,0,0)
        T = np.eye(4)
        T[0:3, 0:3] = 10 * np.eye(3) * (1 / np.max(fuze_trimesh.bounds))
        fuze_trimesh.apply_transform(T)
        
        T = np.eye(4)
        T[0:3, 0:3] = self.rot_x(np.pi / 2)
        fuze_trimesh.apply_transform(T)
        
        T = np.eye(4)
        T[0:3, 0:3] = self.rot_x(np.pi)
        fuze_trimesh.apply_transform(T)
        
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        
        self.scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        self.scene.add(mesh)
        
        # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, zfar=10000, name="cam")
        camera = pyrender.IntrinsicsCamera(
            self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], zfar=10000, name="cam"
        )
        
        s = np.sqrt(2)/2
        camera_pose = np.array([
            [0.0, -s,   s,   0.3],
            [1.0,  0.0, 0.0, 0.0],
            [0.0,  s,   s,   0.35],
            [0.0,  0.0, 0.0, 1.0],
        ])
        
        self.cam_node = self.scene.add(camera, pose=camera_pose)
        light =  pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
        # light = pyrender.SpotLight(color=255 * np.ones(3), intensity=3000.0 * 5, innerConeAngle=np.pi / 16.0)
        self.scene.add(light, pose=camera_pose)
        self.r = pyrender.OffscreenRenderer(self.video_width, self.video_height)
        self.flag = pyrender.constants.RenderFlags.RGBA

    def draw(self, img, rvec, tvec):
        # ===== update cam pose
        camera_pose = np.eye(4)
        res_R, _ = cv2.Rodrigues(rvec)

        # opengl transformation
        # https://stackoverflow.com/a/18643735/4879610
        camera_pose[0:3, 0:3] = res_R.T
        camera_pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        # 180 about x
        camera_pose = camera_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.scene.set_pose(self.cam_node, camera_pose)

        # ====== Render the scene
        color, depth = self.r.render(self.scene, flags=self.flag)
        img[color[:, :, 3] != 0] = color[:, :, 0:3][color[:, :, 3] != 0]
        return img

    def rot_x(self, t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        return m
