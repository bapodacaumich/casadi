import numpy as np
import numpy.linalg as la

class Camera:
    """
    process camera visibility
    """
    def __init__(self, fov, d, gamma=None):
        """
        fov - (horizontal, vertical) in radians
        d - maximum distance from sensor for coverage
        gamma - maximum angle from normal for coverage
        """
        self.hfov = fov[0]
        self.vfov = fov[1]
        self.d = d
        self.gamma = gamma

    def coverage_area(self, pose):
        """
        get bounds of the coverage area for a given pose
        pose - np.array(x, y, z, pan(ccw around z), tilt(positive up)) assume no roll/swing angle
            > Note: pan is bounded by -pi to pi, tilt goes 0 to pi
        """
        # compute rotation matrices
        R_z = np.array([[np.cos(pose[3]), -np.sin(pose[3]), 0.],
                        [np.sin(pose[3]),  np.cos(pose[3]), 0.],
                        [0.,               0.,              1.]])
        R_x = np.array([[1.,              0.,               0.],
                        [0., np.cos(pose[4]), -np.sin(pose[4])],
                        [0., np.sin(pose[4]),  np.cos(pose[4])]])

        dx = self.d*np.sin(self.hfov/2)
        dy = self.d*np.sin(self.vfov/2)
        dz = -np.sqrt(self.d**2 - dx**2 - dy**2)
        tl = R_z @ R_x @ np.array([-dx, dy, dz]) + pose[:3] # top left
        tr = R_z @ R_x @ np.array([dx, dy , dz]) + pose[:3] # top right
        br = R_z @ R_x @ np.array([dx, -dy , dz]) + pose[:3] # bottom right
        bl = R_z @ R_x @ np.array([-dx, -dy , dz]) + pose[:3] # bottom left
        ct = R_z @ R_x @ np.array([0, 0, -self.d]) + pose[:3] # center
        return tl, tr, br, bl, ct

    def coverage(self, pose, point, normal):
        """
        pose - np.array(x, y, z, pan(ccw around z), tilt(positive up)) assume no roll/swing angle
            > Note: pan is bounded by -pi to pi, tilt goes 0 to pi
        point - point on object to observe np.array(x, y, z)
        normal - surface normal for object np.array(x, y, z)

        Camera is aligned with negative Z axis
        """
        # compute viewing vector
        v = point - pose[:3]

        # check point distance
        if la.norm(v) > self.d: return False

        # check normal direction
        if np.dot(v/la.norm(v), normal/la.norm(normal)) > 0: return False

        # compute rotation matrices
        R_z = np.array([[np.cos(-pose[3]), -np.sin(-pose[3]), 0.],
                        [np.sin(-pose[3]),  np.cos(-pose[3]), 0.],
                        [0.              , 0.               , 1.]])
        R_x = np.array([[1.,               0.,                0.],
                        [0., np.cos(-pose[4]), -np.sin(-pose[4])],
                        [0., np.sin(-pose[4]),  np.cos(-pose[4])]])
        v_cam = R_x @ R_z @ v.reshape((3,1))

        # check if point is behind camera
        if v_cam[2] > 0: return False

        # check point in horizontal fov
        h_angle = np.abs(np.arctan(-v_cam[0]/v_cam[2]))
        if h_angle > self.hfov/2:
            # print(pose, point, 'BAD H ANGLE:', h_angle, '/', self.hfov/2)
            return False

        # check point in vertical fov
        v_angle = np.abs(np.arctan(-v_cam[1]/v_cam[2]))
        if v_angle > self.vfov/2:
            # print(pose, point, 'BAD V ANGLE:', v_angle, '/', self.vfov/2)
            return False

        # TODO: DEBUG THIS, disregard surface angle for now!
        # # check if surface angle is good
        # surface_angle = np.arccos(np.dot(-v, normal)/(la.norm(-v) * la.norm(normal)))
        # if surface_angle > self.gamma: return False
        return True
