import numpy as np

K_list = np.load("Video_Camera_Processing/extrinsics_tune2_K_list.npy", allow_pickle=True)
R_list = np.load("Video_Camera_Processing/extrinsics_tune2_R_list.npy", allow_pickle=True)
t_list = np.load("Video_Camera_Processing/extrinsics_tune2_t_list.npy", allow_pickle=True)
P_list = []
for K, R, t in zip(K_list, R_list, t_list):
    P = K @ np.hstack([R, t])
    P_list.append(P)
print(P_list)

np.save("Video_Camera_Processing/P_list.npy", P_list)

