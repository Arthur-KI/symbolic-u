from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image
try:
 import cv2
except Exception:
 cv2=None

def observe_image(path:str):
 arr=np.array(Image.open(path).convert('RGB'))
 h,w,_=arr.shape
 border=np.concatenate([arr[0,:,:],arr[-1,:,:],arr[:,0,:],arr[:,-1,:]],axis=0)
 bg=np.median(border,axis=0)
 dist=np.linalg.norm(arr.astype(float)-bg[None,None,:],axis=2)
 mask=(dist>25).astype(np.uint8)*255
 out={'image_size':[int(w),int(h)],'background_rgb':[float(x) for x in bg],'regions':[]}
 if cv2 is None:
  out['status']='opencv_missing'; return out
 n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,connectivity=8)
 for lab in range(1,n):
  area=int(stats[lab,cv2.CC_STAT_AREA])
  if area<200: continue
  component=(labels==lab).astype(np.uint8)*255
  contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if not contours: continue
  cnt=max(contours,key=cv2.contourArea)
  peri=cv2.arcLength(cnt,True)
  approx=cv2.approxPolyDP(cnt,0.03*peri,True)
  ys,xs=np.where(labels==lab)
  mean_rgb=arr[ys,xs,:].mean(axis=0)
  out['regions'].append({'area':area,'centroid':[float(centroids[lab][0]),float(centroids[lab][1])],'mean_rgb':[float(x) for x in mean_rgb],'corner_count_raw':int(len(approx))})
 out['region_count']=len(out['regions']); out['status']='ok'; return out

if __name__=='__main__':
 import sys
 for fn in sys.argv[1:]: print(json.dumps({'file':fn,'obs':observe_image(fn)},indent=2))
