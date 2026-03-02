import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P_list_path = "Video_Camera_Processing/P_list.npy"
K_list_path = "Video_Camera_Processing/K_list.npy"
R_list_path = "Video_Camera_Processing/R_list.npy"
t_list_path = "Video_Camera_Processing/t_list.npy"
plot_path = "Video_Camera_Processing/camera_poses.png"


# P (projects 3D world points to image pixels) = K @ [R | t]
# K : intrinsics (fx, fy, cx, cy) in PIXEL units for a specific image size
#   - fx, fy : focal lengths (pixels)
#   - cx, cy : principal point (pixels)
# R : rotation matrix
# t : camera position (translation vector)


def ypr_to_rotation(yaw, pitch, roll):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp,0, cp]
    ])

    Rx = np.array([
        [1, 0,  0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    return Rz @ Ry @ Rx

def make_P(fx, fy, cx, cy, R, t):
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t_pos = np.asarray(t, dtype=np.float64).reshape(3, 1)
    t_proj = -R @ t_pos
    Rt = np.hstack([R, t_proj])
    return K @ Rt, K, R, t_pos

def make_P2(K, R, t):
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R, t])
    return K @ Rt, K, R, t


def look_at(camera_position, target=np.array([0.0, 0.0, 0.0]), world_up=np.array([0.0, 0.0, 1.0])):
    C = np.asarray(camera_position, dtype=np.float64).ravel()
    T = np.asarray(target, dtype=np.float64).ravel()
    forward = T - C
    n = np.linalg.norm(forward)
    if n < 1e-10:
        raise ValueError("Camera position and target are too close.")
    forward = forward / n
    right = np.cross(forward, world_up)
    nr = np.linalg.norm(right)
    if nr < 1e-10:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / nr
    up = np.cross(right, forward)
    R = np.column_stack([right, up, forward]).T
    return R


def camera_center_and_view(R, t):
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    C = t.ravel() if t.size == 3 else t.reshape(3)
    view_dir = R.T @ np.array([0.0, 0.0, 1.0])
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-10)
    return C, view_dir


# If cameras look at the origin, set USE_LOOK_AT = True
USE_LOOK_AT = True
TARGET_LOOK_AT = np.array([0.0, 0.0, 2.0])

cameras = [
    {
        "fx": 1100.0, "fy": 1100.0, "cx": 960.0, "cy": 720.0,
        "R": ypr_to_rotation(-3.44628506803, 0, 0),
        "t": [23.8485, 7.5, 2.0],
    },
    {
        "fx": 4800.0, "fy": 4800.0, "cx": 1920.0, "cy": 1080.0,
        "R": ypr_to_rotation(3.44628506803, 0, 0),
        "t": [23.8485, -7.5, 2.0],
    }
]

if USE_LOOK_AT:
    for i, c in enumerate(cameras):
        C = np.asarray(c["t"], dtype=np.float64).ravel()
        R_new = look_at(C, TARGET_LOOK_AT)
        cameras[i]["R"] = R_new

P_list, K_list, R_list, t_list = zip(*[make_P(c["fx"], c["fy"], c["cx"], c["cy"], c["R"], c["t"]) for c in cameras])
arr = np.stack(P_list)
np.save(P_list_path, arr)
np.save(K_list_path, K_list)
np.save(R_list_path, R_list)
np.save(t_list_path, t_list)
print(f"Saved P_list: {P_list}")
print(f"Saved K_list: {K_list}")
print(f"Saved R_list: {R_list}")
print(f"Saved t_list: {t_list}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
arrow_scale = 5.0
ax.quiver(0, 0, 0, 1, 0, 0, color="r", arrow_length_ratio=0.15, linewidth=2, label="X")
ax.quiver(0, 0, 0, 0, 1, 0, color="g", arrow_length_ratio=0.15, linewidth=2, label="Y")
ax.quiver(0, 0, 0, 0, 0, 1, color="b", arrow_length_ratio=0.15, linewidth=2, label="Z")
colors = ["orange", "cyan"]
all_pts = [[0, 0, 0]]
for i, (R, t) in enumerate(zip(R_list, t_list)):
    C, view_dir = camera_center_and_view(R, t)
    all_pts.append(C)
    all_pts.append(C + view_dir * arrow_scale)
    ax.scatter(*C, color=colors[i], s=80, label=f"Cam {i}")
    ax.quiver(C[0], C[1], C[2], view_dir[0] * arrow_scale, view_dir[1] * arrow_scale, view_dir[2] * arrow_scale, color=colors[i], arrow_length_ratio=0.2, linewidth=2)
all_pts = np.array(all_pts)
margin = 2.0
max_axis = float(np.abs(all_pts).max()) + margin
max_axis = max(max_axis, 10.0)
ax.set_xlim(-max_axis, max_axis)
ax.set_ylim(-max_axis, max_axis)
ax.set_zlim(-max_axis, max_axis)
ax.legend()
ax.set_title("Camera positions and viewing direction")
plt.tight_layout()
plt.savefig(plot_path, dpi=120)
print(f"Saved camera pose plot to {plot_path}")
plt.show()
