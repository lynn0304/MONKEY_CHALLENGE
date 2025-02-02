from stardist.models import StarDist2D
import cv2
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE
    
def dilatation(src, size, val=2):
    dilation_shape = morph_shape(val)
    element = cv2.getStructuringElement(dilation_shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.dilate(src, element)

cell_type = [
    # "inflammatory-cells",
    "lymphocytes",
    "monocytes",
]
cell_color = {
    "inflammatory-cells":   (0,255,0),
    "lymphocytes":          (255,0,0),
    "monocytes":            (0,0,255),
}
cell_size = {
    "inflammatory-cells":   3,
    "lymphocytes":          2,
    "monocytes":            1,
}
# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_he')
# model = StarDist2D(None, name='stardist', basedir='models')
root = r'/home/ubuntu/datadisk/MONKEY/train_yolo/train_roi/'
annotation_dir = r'/home/ubuntu/datadisk/MONKEY/annotations_1115/json/'
vis_dir = r'/home/ubuntu/datadisk/MONKEY/train_yolo/train_yolo/vis/'
tile_dir = r'/home/ubuntu/datadisk/MONKEY/train_yolo/train_yolo/images/train/'
label_dir = r'/home/ubuntu/datadisk/MONKEY/train_yolo/train_yolo/labels/train/'
roi_count = 0
for path in os.listdir(root):
    # print(roi_count, "is starting")
    if path[-4:] != '.png': continue
    # if roi_count < 27: 
    #     roi_count+=1
    #     continue
    # path = 'A_P000003_19854_46484_21152_47573.png'
    prefix = "_".join(path.split("_")[:2])
    left = int(path.split("_")[2])
    top = int(path.split("_")[3])
    right = int(path.split("_")[4])
    button = int(path.split("_")[5].split(".")[0])
    print(root+path)
    img = cv2.imread(root+path)[:,:,::-1]
    img = np.array(img)
    train_img = img.copy()
    print(left, top, right, button)

    contour = None
    for ct in cell_type:
        with open(annotation_dir+prefix+"_"+ct+".json", 'rb') as f:
            json_dict = json.load(f)
            for roi in json_dict['rois']:
                con = np.array(roi['polygon'], dtype=np.int32)
                # con = con.transpose(1,0)
                con = con[:,None,:]
                # print(con.shape)
                M = cv2.moments(con)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if (left <= cX and cX <= right) and (top <= cY and cY <= button):
                    con[:,:,0] = con[:,:,0] - left
                    con[:,:,1] = con[:,:,1] - top
                    contour = con
                # print(roi['polygon'])
    prob_thresh = 0.2
    if img.shape[0]*img.shape[1] >= 3920*3920:
        print("big image")
        _, ret = model.predict_instances_big(img.copy()/255, axes='YXC', block_size=2048, min_overlap=128, prob_thresh=prob_thresh, nms_thresh=0.3)
    else:
        _, ret = model.predict_instances(img.copy()/255, prob_thresh=prob_thresh, nms_thresh=0.3)
    coord = ret['coord']
    cells = {
        "inflammatory-cells": [],
        "lymphocytes": [],
        "monocytes": [],
    }
    take_cell = np.zeros((coord.shape[0],), dtype=np.uint8)
    mask = np.zeros_like(img[:,:,0])
    labels = mask.copy().astype(np.int32)
    for i, co in enumerate(coord):
        co = co.transpose(1,0)[:,None,::-1].astype(np.int32)
        cell = mask.copy()
        cell = cv2.drawContours(cell, [co], -1, (255,), -1)
        labels[cell!=0] = i+1
    
    for ct in cell_type:
        with open(annotation_dir+prefix+"_"+ct+".json", 'rb') as f:
            json_dict = json.load(f)
            for p in json_dict['points']:
                x, y = p['point']
                if not ((left <= x and x <= right) and (top <= y and y <= button)): continue

                x = int(x)-left
                y = int(y)-top
                # print(x, y)
                # print(labels.shape)
                if labels[y, x] != 0:
                    cell_label = (labels==labels[y, x]).astype(np.uint8)*255
                    mask[labels==labels[y, x]] = 255
                    idx = labels[y, x]
                    take_cell[idx-1] = 1
                    labels[labels==labels[y, x]] = 0
                    co = coord[idx-1]
                    co = co.transpose(1,0)[:,None,::-1].astype(np.int32)
                    mask = cv2.drawContours(mask, [co], -1, (0,), -1)
                    
                    con, hir = cv2.findContours(cell_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cells[ct].append({
                        "type": "cell_hit",
                        "contour": con,
                    })
                    # print('hit')
                else:
                    cells[ct].append({
                        "type": "cell_miss",
                        "contour": np.array([x, y], dtype=np.int32),
                    })
                    # print('miss')

                # print(p['point'])
            # print(json_dict['points'])
            # print(json_dict.keys())

    

    mask = cv2.drawContours(mask, [contour], -1, (255,), -1)
    mask = dilatation(mask, 3)
    mask = cv2.GaussianBlur(mask, (15, 15),5)

    # plt.imshow(mask)
    # plt.show()
    bounding_boxes = []
    for k in cell_type:
        v = cells[k]
        for cell in v:
            if cell['type'] == "cell_hit":
                # print(cell['contour'])
                img = cv2.drawContours(img, cell['contour'], -1, cell_color[k], cell_size[k])    
                boundRect = cv2.boundingRect(cell['contour'][0])
                cv2.rectangle(img, (int(boundRect[0]), int(boundRect[1])), \
                    (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), cell_color[k], cell_size[k])
                bounding_boxes.append([boundRect[0]+boundRect[2]/2,boundRect[1]+boundRect[3]/2,boundRect[2],boundRect[3], cell_type.index(k)])
            elif cell['type'] == "cell_miss":
                img = cv2.circle(img, cell['contour'], 10, (255,0,255), 2)    
    for i, tc in enumerate(take_cell):
        if tc == 0:
            co = coord[i]
            co = co.transpose(1,0)[:,None,::-1].astype(np.int32)
            img = cv2.drawContours(img, [co], -1, (0,255,0), 1)    
            boundRect = cv2.boundingRect(co)
            bounding_boxes.append([boundRect[0]+boundRect[2]/2, boundRect[1]+boundRect[3]/2,boundRect[2],boundRect[3], -1])

    bounding_boxes = np.array(bounding_boxes)
    print(bounding_boxes.shape)

    mask = (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255)
    inv_img = np.ones_like(mask)*255
    img = img*mask
    inv_img = inv_img*(1-mask)
    img = (img+inv_img).astype(np.uint8)
    print(ret.keys())
    print(labels.shape)
    # print(vis_dir+'/'+prefix+".tif")
    cv2.imwrite(vis_dir+prefix+"_"+"_".join([str(left), str(top), str(right), str(button), ])+".tif", img[:,:,::-1])
    patch_size = 416
    tile_w = int(math.ceil(train_img.shape[1]/patch_size))
    tile_h = int(math.ceil(train_img.shape[0]/patch_size))
    tmp = np.ones((tile_h*patch_size, tile_w*patch_size, 3), dtype=np.uint8)*255
    tmp_mask = np.zeros((tile_h*patch_size, tile_w*patch_size), dtype=mask.dtype)

    train_img = train_img*mask
    train_img = (train_img+inv_img).astype(np.uint8)

    tmp[:train_img.shape[0], :train_img.shape[1], :] = train_img
    train_img = tmp
    tmp_mask[:mask.shape[0], :mask.shape[1]] = mask[:,:,0]
    mask = tmp_mask
    # half_patch = patch_size//2
    for j in range(tile_h):
        for i in range(tile_w):
            patch = train_img[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size, :]
            patch_mask = mask[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size]
            tile_obj = ((i*patch_size<=bounding_boxes[:,0]) & (bounding_boxes[:,0] < (i+1)*patch_size)) & \
                        ((j*patch_size<=bounding_boxes[:,1]) & (bounding_boxes[:,1] < (j+1)*patch_size))
            # print(bounding_boxes[:,4])
            labels = []
            cell_count = 0
            for box in bounding_boxes[tile_obj]:
                color = cell_color[cell_type[int(box[4])]]
                if box[4]<0: color = (0,255,0)
                labels_num = box[4]
                if labels_num < 0: labels_num = len(cell_type)
                box[0] = box[0] - i*patch_size
                box[1] = box[1] - j*patch_size
                if patch_mask[int(box[1]), int(box[0])] != 1: continue
                normal = box[:-1]/patch_size
                if normal[0]>1: print("out normal", normal[0])
                if normal[1]>1: print("out normal", normal[1])
                # patch = cv2.rectangle(patch, (int(x-box[2]/2), int(y-box[3]/2)), \
                #             (int(x+box[2]/2), int(y+box[3]/2)), color, 1)
                labels.append([int(labels_num)]+(normal).tolist())
                cell_count+=1
            if cell_count <= 0: continue
            
            
            cv2.imwrite(tile_dir+prefix+"_"+"_".join([str(left), str(top), str(right), str(button), str(i*patch_size), str(j*patch_size)])+".tif", patch[:,:,::-1])
            with open(label_dir+prefix+"_"+"_".join([str(left), str(top), str(right), str(button), str(i*patch_size), str(j*patch_size)])+".txt", 'w') as f:
                for l in labels:
                    f.write( " ".join(map(str, l) )+'\n')
            # print(cell_count)
            # plt.imshow(patch)
            # plt.show()
            
    roi_count+=1

    # plt.imshow(img)
    # plt.show()