"""ROS PointCloud2 conversion helpers kept outside the exploration logic."""

from typing import Optional, Tuple

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


FIELDS_XYZI = (
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
)


def create_xyzi_cloud(
    stamp,
    frame_id: str,
    points: np.ndarray,
    hit_mask: Optional[np.ndarray] = None,
) -> PointCloud2:
    values = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if hit_mask is None:
        intensity = np.ones((len(values), 1), dtype=np.float32)
    else:
        intensity = np.asarray(hit_mask, dtype=np.float32).reshape((-1, 1))
    xyzi = np.concatenate((values, intensity), axis=1)
    return point_cloud2.create_cloud(
        Header(stamp=stamp, frame_id=frame_id), FIELDS_XYZI, xyzi
    )


def read_xyzi_cloud(message: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
    names = {field.name for field in message.fields}
    requested = ["x", "y", "z"]
    has_intensity = "intensity" in names
    if has_intensity:
        requested.append("intensity")
    values = point_cloud2.read_points_numpy(
        message, field_names=requested, skip_nans=True
    )
    matrix = np.asarray(values)
    if matrix.dtype.fields:
        columns = [matrix[name].astype(np.float32) for name in requested]
        matrix = np.column_stack(columns)
    matrix = np.asarray(matrix, dtype=np.float32).reshape((-1, len(requested)))
    points = matrix[:, :3]
    hit = matrix[:, 3] > 0.5 if has_intensity else np.ones(len(points), dtype=bool)
    return points, hit
