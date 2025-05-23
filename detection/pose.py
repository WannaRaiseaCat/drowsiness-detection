import cv2
import numpy as np


class HeadPose:
    def __init__(self, faceMesh, cameraMatrix=None, distCoeffs=None):
        self.faceMesh = faceMesh
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.rvec = None
        self.tvec = None

    def process_image(self, frame):
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.faceMesh.process(image)
        image.flags.writeable = True
        self.frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.imgH, self.imgW = self.frame.shape[:2]

        if self.cameraMatrix is None:
            self.size = self.frame.shape
            self.focalLength = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.cameraMatrix = np.array([
                [self.focalLength, 0, self.center[0]],
                [0, self.focalLength, self.center[1]],
                [0, 0, 1]
            ], dtype="double")

        if self.distCoeffs is None:
            self.distCoeffs = np.zeros((4, 1))

        return self.frame, results

    def estimate_pose(self, frame, results, display=False):
        self.frame = frame
        self.face3d, self.face2d = [], []
        self.nose2d, self.nose3d = (0, 0), (0.0, 0.0, 0.0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * self.imgW), int(lm.y * self.imgH)
                        if idx == 1:
                            self.nose2d = (x, y)
                            self.nose3d = (lm.x * self.imgW, lm.y * self.imgH, lm.z * self.imgW)

                        self.face2d.append([x, y])
                        self.face3d.append([x, y, lm.z])

                self.face2d = np.array(self.face2d, dtype=np.float64)
                self.face3d = np.array(self.face3d, dtype=np.float64)

                # Robust solve with RANSAC
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=self.face3d,
                    imagePoints=self.face2d,
                    cameraMatrix=self.cameraMatrix,
                    distCoeffs=self.distCoeffs,
                    reprojectionError=8.0,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not success:
                    continue

                try:
                    rvec, tvec = cv2.solvePnPRefineVVS(
                        self.face3d, self.face2d,
                        self.cameraMatrix, self.distCoeffs,
                        rvec, tvec
                    )
                except AttributeError:
                    pass

                self.rvec, self.tvec = rvec, tvec

                if display:
                    self.calculate_angles()
                    self.display_direction()

        return self.frame

    def calculate_angles(self):
        rmat = cv2.Rodrigues(self.rvec)[0]
        angles = cv2.RQDecomp3x3(rmat)[0]
        # Convert to degrees
        self.roll = angles[0] * 360
        self.pitch = angles[1] * 360
        self.yaw = angles[2] * 360
        return self.roll, self.pitch, self.yaw

    def _draw_axes(self, axis_length=50):
        origin = np.array(self.nose3d, dtype=np.float32).reshape((1, 3))
        axes = np.float32([
            [axis_length, 0, 0],  # X-axis
            [0, axis_length, 0],  # Y-axis
            [0, 0, axis_length]   # Z-axis
        ])
        imgpts, _ = cv2.projectPoints(
            np.vstack((origin, origin + axes)),
            self.rvec, self.tvec,
            self.cameraMatrix, self.distCoeffs
        )
        origin_pt = tuple(imgpts[0].ravel().astype(int))
        x_pt, y_pt, z_pt = [tuple(pt.ravel().astype(int)) for pt in imgpts[1:4]]

        cv2.line(self.frame, origin_pt, x_pt, (0, 0, 255), 2)
        cv2.line(self.frame, origin_pt, y_pt, (0, 255, 0), 2)
        cv2.line(self.frame, origin_pt, z_pt, (255, 0, 0), 2)

        cv2.putText(self.frame, f"R:{self.roll:.1f}", (origin_pt[0] + 5, origin_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(self.frame, f"P:{self.pitch:.1f}", (origin_pt[0] + 5, origin_pt[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(self.frame, f"Y:{self.yaw:.1f}", (origin_pt[0] + 5, origin_pt[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    def _draw_nose_vector(self):
        # Draw a vector from nose to indicate pitch and roll direction
        p1 = (int(self.nose2d[0]), int(self.nose2d[1]))
        p2 = (int(self.nose2d[0] + self.pitch * 10), int(self.nose2d[1] - self.roll * 10))
        cv2.line(self.frame, p1, p2, (0, 0, 255), 3)
        # Overlay roll, pitch, yaw text
        cv2.putText(self.frame, f"Roll: {self.roll:.2f}", (p1[0] + 5, p1[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(self.frame, f"Pitch: {self.pitch:.2f}", (p1[0] + 5, p1[1] + 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(self.frame, f"Yaw: {self.yaw:.2f}", (p1[0] + 5, p1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def display_direction(self):
        self._draw_axes()
        self._draw_nose_vector()
        return self.frame
