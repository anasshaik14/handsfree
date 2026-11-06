"""MediaPipe FaceMesh landmark indices used across modalities.

Naming is from the SUBJECT's point of view. In a non-mirrored camera frame the
subject's RIGHT eye appears on the image LEFT.
"""

# 6-point EAR sets (p1..p6 in the Soukupova & Cech ordering)
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Iris centers (FaceMesh refine_landmarks=True adds 468..477)
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_RING = [469, 470, 471, 472]
LEFT_IRIS_RING = [474, 475, 476, 477]

# Eye box corners: (outer, inner, top, bottom)
RIGHT_EYE_BOX = dict(outer=33, inner=133, top=159, bottom=145)
LEFT_EYE_BOX = dict(inner=362, outer=263, top=386, bottom=374)

# Eyelid points for the Eye Lid Angle (ELA) measure
RIGHT_LID = dict(corner=33, upper=(160, 158), lower=(144, 153))
LEFT_LID = dict(corner=263, upper=(387, 385), lower=(373, 380))

# solvePnP correspondence points
PNP_NOSE = 1
PNP_CHIN = 152
PNP_RIGHT_EYE_OUTER = 33     # subject right -> image left
PNP_LEFT_EYE_OUTER = 263
PNP_MOUTH_RIGHT = 61         # subject right mouth corner
PNP_MOUTH_LEFT = 291

# Rigid subset used by the Kabsch (mp_geometry) head-pose provider —
# avoids the jaw/lips which deform with speech.
RIGID_SUBSET = [1, 4, 5, 6, 8, 9, 10, 33, 133, 152, 168, 234, 263, 362, 454]

# Face extremes for the heuristic provider
FACE_RIGHT_EDGE = 234        # subject right cheek (image left)
FACE_LEFT_EDGE = 454
FOREHEAD = 10
CHIN = 152
NOSE_TIP = 1
BETWEEN_EYES = 168

# Mouth points for the yawn Mouth Aspect Ratio (fatigue detector):
# MAR = |13-14| (inner lip gap) / |61-291| (mouth width)
MOUTH_UPPER_INNER = 13
MOUTH_LOWER_INNER = 14
MOUTH_CORNER_RIGHT = 61      # == PNP_MOUTH_RIGHT
MOUTH_CORNER_LEFT = 291      # == PNP_MOUTH_LEFT
