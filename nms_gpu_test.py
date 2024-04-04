import torch
import time
import nms_gpu
import numpy as np
bbox=np.load("bbox.npy")
print("bbox shape: ", bbox.shape)
start = time.perf_counter()
n_bbox = bbox.shape[0]
keep_out=np.zeros(bbox.shape[0],dtype=np.int32)
n=nms_gpu.nms_gpu(keep_out,bbox)
end = time.perf_counter()
print("nms_gpu time: ", end - start)
print(n , keep_out)

start = time.perf_counter()
bbox_t = torch.from_numpy(bbox).cuda()
scores = torch.rand(bbox_t.shape[0]).cuda()
nms_overlap_thresh = 0.7
result = nms_gpu.nms_gpu_tensor(bbox_t, scores, nms_overlap_thresh)
end = time.perf_counter()
print("nms_gpu_tensor time: ", end - start)
print(result)