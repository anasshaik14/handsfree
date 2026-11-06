"""Head pose providers: OpenCV solvePnP, MediaPipe geometry (Kabsch), and
landmark-ratio heuristics.

All providers output HeadPose with a shared sign convention derived from the
face-forward vector (no Euler-order ambiguity):
    yaw   > 0 : turned to the user's LEFT
    pitch > 0 : tilted UP
Neutral is subtracted per user (auto-captured at start, re-settable live).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import cv2
import numpy as np

from app.core.interfaces import HeadPoseProvider
from app.core.types import FaceResult, HeadPose
from app.vision import landmarks as L


def _angles_from_forward(fwd: np.ndarray, up: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll (deg) from face-forward and head-up vectors in camera
    coordinates (x right, y down, z into the scene).

    A face looking straight at the camera has fwd ~ (0, 0, -1), up ~ (0, -1, 0).
    Turning left swings fwd toward +x; looking up swings it toward -y.
    """
    yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
    pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))
    roll = math.degrees(math.atan2(up[0], -up[1]))
    return yaw, pitch, roll


class _NeutralOffset:
    """Auto-captures a neutral pose from the first stable second of samples.

    Also re-centers slowly while the head is near neutral: posture shifts
    (slouching, leaning) otherwise accumulate as a permanent angle bias that
    makes small movements read as large ones.
    """

    ADAPT_RATE = 0.01          # per frame (~3 s time constant at 30 fps)
    ADAPT_BAND = (6.0, 5.0, 6.0)   # only adapt while |residual| is this small

    def __init__(self, warmup: int = 25) -> None:
        self._buf: deque = deque(maxlen=warmup)
        self.offset: Optional[tuple[float, float, float]] = None

    def apply(self, yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
        if self.offset is None:
            self._buf.append((yaw, pitch, roll))
            if len(self._buf) == self._buf.maxlen:
                arr = np.array(self._buf)
                self.offset = tuple(np.median(arr, axis=0))
            return 0.0, 0.0, 0.0  # not calibrated yet -> report neutral
        oy, op, orr = self.offset
        ry, rp, rr = yaw - oy, pitch - op, roll - orr
        by, bp, br = self.ADAPT_BAND
        if abs(ry) < by and abs(rp) < bp and abs(rr) < br:
            k = self.ADAPT_RATE
            self.offset = (oy + ry * k, op + rp * k, orr + rr * k)
        return ry, rp, rr

    def set_from(self, yaw: float, pitch: float, roll: float) -> None:
        self.offset = (yaw, pitch, roll)

    def reset(self) -> None:
        self._buf.clear()
        self.offset = None


class SolvePnPHeadPose(HeadPoseProvider):
    """Classic 6-point PnP against a generic 3D face model (Mallick model)."""

    name = "solvepnp"

    _MODEL = np.array([
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (225.0, 170.0, -135.0),   # subject-right eye outer (image left)
        (-225.0, 170.0, -135.0),  # subject-left eye outer
        (150.0, -150.0, -125.0),  # subject-right mouth corner
        (-150.0, -150.0, -125.0), # subject-left mouth corner
    ], dtype=np.float64)
    _IDX = [L.PNP_NOSE, L.PNP_CHIN, L.PNP_RIGHT_EYE_OUTER,
            L.PNP_LEFT_EYE_OUTER, L.PNP_MOUTH_RIGHT, L.PNP_MOUTH_LEFT]

    def __init__(self) -> None:
        self._neutral = _NeutralOffset()
        self._last = (0.0, 0.0, 0.0)

    def process(self, face: FaceResult) -> Optional[HeadPose]:
        if not face.ok:
            return None
        pts2d = np.array([face.landmarks[i][:2] for i in self._IDX], dtype=np.float64)
        w, h = face.width, face.height
        cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(self._MODEL, pts2d, cam, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        # Model: x -> subject-right appears image-left... model +x was defined
        # toward image RIGHT for the subject-right point? No: subject-right eye
        # (image left) has model x=+225 while its image x is small, so the fit
        # makes model +x map to image -x; the forward axis still disambiguates:
        # model z is negative into the head, so face-forward = model +z.
        fwd = R @ np.array([0.0, 0.0, 1.0])
        up = R @ np.array([0.0, 1.0, 0.0])
        # Model y is up while image y is down; the PnP solution absorbs this
        # as a 180-deg x-rotation, so 'up' comes out toward image -y already.
        yaw, pitch, roll = _angles_from_forward(fwd, -up if up[1] > 0 else up)
        # Empirical convention fix: with this model, turning left (nose toward
        # image +x) yields fwd_x < 0 because model x is mirrored vs image x.
        yaw = -yaw
        y, p, r = self._neutral.apply(yaw, pitch, roll)
        self._last = (yaw, pitch, roll)
        return HeadPose(t=face.t, yaw=y, pitch=p, roll=r,
                        valid=self._neutral.offset is not None)

    def set_neutral(self, face: FaceResult) -> bool:
        self._neutral.set_from(*self._last)
        return True

    def reset(self) -> None:
        self._neutral.reset()


class MPGeometryHeadPose(HeadPoseProvider):
    """MediaPipe Face Geometry approach: rigid (Kabsch/SVD) alignment of the
    live metric landmark cloud against the user's own neutral cloud.

    Personalizes to the actual face shape instead of a generic model, which
    removes the anthropometric error term of generic PnP.
    """

    name = "mp_geometry"

    def __init__(self) -> None:
        self._neutral_cloud: Optional[np.ndarray] = None
        self._auto_buf: deque = deque(maxlen=25)
        self._last_cloud: Optional[np.ndarray] = None

    @staticmethod
    def _cloud(face: FaceResult) -> np.ndarray:
        pts = face.landmarks[L.RIGID_SUBSET].copy()
        pts -= pts.mean(axis=0)
        norm = np.sqrt((pts ** 2).sum(axis=1).mean())
        return pts / norm if norm > 1e-9 else pts

    @staticmethod
    def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Rotation R minimizing ||R@a - b||."""
        H = a.T @ b
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        return Vt.T @ D @ U.T

    def process(self, face: FaceResult) -> Optional[HeadPose]:
        if not face.ok:
            return None
        cloud = self._cloud(face)
        self._last_cloud = cloud
        if self._neutral_cloud is None:
            self._auto_buf.append(cloud)
            if len(self._auto_buf) == self._auto_buf.maxlen:
                self._neutral_cloud = np.mean(np.stack(self._auto_buf), axis=0)
            return HeadPose(t=face.t, yaw=0.0, pitch=0.0, roll=0.0, valid=False)
        R = self._kabsch(self._neutral_cloud, cloud)
        # Neutral face-forward in camera coords is toward the camera: (0,0,-1).
        fwd = R @ np.array([0.0, 0.0, -1.0])
        up = R @ np.array([0.0, -1.0, 0.0])   # image y is down; head-up = -y
        yaw, pitch, roll = _angles_from_forward(fwd, up)
        # Near neutral, let the reference cloud track slow posture changes so
        # they never accumulate into an angle bias.
        if abs(yaw) < 6.0 and abs(pitch) < 5.0 and abs(roll) < 6.0:
            self._neutral_cloud = self._neutral_cloud * 0.99 + cloud * 0.01
        return HeadPose(t=face.t, yaw=yaw, pitch=pitch, roll=roll, valid=True)

    def set_neutral(self, face: FaceResult) -> bool:
        if self._last_cloud is None:
            return False
        self._neutral_cloud = self._last_cloud.copy()
        return True

    def reset(self) -> None:
        self._neutral_cloud = None
        self._auto_buf.clear()


class HeuristicHeadPose(HeadPoseProvider):
    """Landmark-ratio heuristics: cheapest method, no 3D math at all.

    yaw   : nose offset between the two cheek edges
    pitch : nose vertical position between eye line and chin
    roll  : eye-line angle
    """

    name = "landmark_heuristic"

    YAW_SCALE = 90.0    # deg per unit of normalized nose offset
    PITCH_SCALE = 160.0

    def __init__(self) -> None:
        self._neutral = _NeutralOffset()
        self._last = (0.0, 0.0, 0.0)

    def process(self, face: FaceResult) -> Optional[HeadPose]:
        if not face.ok:
            return None
        pts = face.landmarks
        nose = pts[L.NOSE_TIP][:2]
        right_edge = pts[L.FACE_RIGHT_EDGE][:2]   # image left
        left_edge = pts[L.FACE_LEFT_EDGE][:2]     # image right
        eye_mid = (pts[L.PNP_RIGHT_EYE_OUTER][:2] + pts[L.PNP_LEFT_EYE_OUTER][:2]) / 2.0
        chin = pts[L.CHIN][:2]

        face_w = np.linalg.norm(left_edge - right_edge)
        if face_w < 1e-6:
            return None
        d_right = np.linalg.norm(nose - right_edge)
        d_left = np.linalg.norm(nose - left_edge)
        # Turning left moves the nose toward the subject-left edge (image right):
        # d_left shrinks -> positive yaw.
        yaw = (d_right - d_left) / face_w * self.YAW_SCALE

        face_h = np.linalg.norm(chin - eye_mid)
        r = float(np.dot(nose - eye_mid, chin - eye_mid) / (face_h ** 2 + 1e-9))
        # Looking up raises the nose toward the eye line -> r shrinks -> pitch up.
        pitch = (0.52 - r) * self.PITCH_SCALE

        eye_vec = pts[L.PNP_LEFT_EYE_OUTER][:2] - pts[L.PNP_RIGHT_EYE_OUTER][:2]
        roll = -math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

        y, p, rr = self._neutral.apply(yaw, pitch, roll)
        self._last = (yaw, pitch, roll)
        return HeadPose(t=face.t, yaw=y, pitch=p, roll=rr,
                        valid=self._neutral.offset is not None)

    def set_neutral(self, face: FaceResult) -> bool:
        self._neutral.set_from(*self._last)
        return True

    def reset(self) -> None:
        self._neutral.reset()
