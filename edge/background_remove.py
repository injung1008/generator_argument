from imantics import Polygons, Mask
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pickle
import os
import random
from rembg import remove
from rembg import new_session
import copy
import json
import shutil

class Make_Custom_Dataset():
    def __init__(self):
        self.train_data_r = 0.8
        self.id_cnt = 0
        self.yolo_label = True
        self.mask_dict = {}
        self.resize = True
        self.rotation = True
        
    
    
    
    def remove_background(self, img, only_mask=True, post_process_mask=True,model='u2net'):
        
        out = remove(img, only_mask=only_mask,post_process_mask=post_process_mask,session=new_session(model))
        mask = out.copy()
                
        return mask
    
    def fit_bbox(self, target_img, mask, back_img):
        
        x, y, w, h = cv2.boundingRect(mask)
        mask[mask==0] = 100
        mask[mask==255] = 0
        mask[mask==100] = 255
        mask = mask[y:y+h,x:x+w]
        target_img = target_img[y:y+h,x:x+w]
        
        resize_size_w = mask.shape[1]
        resize_size_h = mask.shape[0]
        
        x_limit = int(back_img.shape[1])-resize_size_w
        y_limit = int(back_img.shape[0])-resize_size_h
        
        if x_limit <= 10 or y_limit <= 10 : 
            print(f'back_img : {back_img.shape}')
            print(f'resize_size_w : {resize_size_w} resize_size_h : {resize_size_h}')
            print(f'x_limit : {x_limit} y_limit : {y_limit}')
            back_img, mask, target_img = self.resize_img(back_img, mask, target_img)
            print(f'mask : {mask.shape}')
            print(f'back_img : {back_img.shape}')
        
        return target_img, mask, back_img
                
    
    
    
    def resize_img(self, back_img, mask, target_img):
        resize_size = random.randrange(1,3)

        if back_img.shape[1]-50 <= mask.shape[1] or back_img.shape[0]-50 <= mask.shape[0] :
            w_check = mask.shape[1]/back_img.shape[1]
            h_check = mask.shape[0]/back_img.shape[0]
            resize_size = round(max(w_check, h_check),1)*1.5
            print(f'resize_size : {resize_size}')

        resize_size_w = int(mask.shape[1]/resize_size)
        resize_size_h = int(mask.shape[0]/resize_size)

        mask = cv2.resize(mask, (resize_size_w, resize_size_h), interpolation=cv2.INTER_NEAREST)
        target_img = cv2.resize(target_img, (resize_size_w, resize_size_h), interpolation=cv2.INTER_NEAREST)

        
        return back_img, mask, target_img
        
    
    
    
    def rotation_img(self, target_img, mask):

    
        # 이미지의 크기와 회전 중심점을 설정합니다.
        t_height, t_width = target_img.shape[:2]
        t_center = (t_width/2, t_height/2)
        
        # 회전 각도와 변환 행렬을 계산합니다.
        r_degree = random.randrange(0,360)
        M = cv2.getRotationMatrix2D(t_center, r_degree,1)
        
        # 회전된 이미지의 크기를 계산합니다.
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_width = int((t_height * sin) + (t_width * cos))
        new_height = int((t_height * cos) + (t_width * sin))

        # 새로운 이미지를 생성합니다.
        M[0, 2] += (new_width / 2) - t_center[0]
        M[1, 2] += (new_height / 2) - t_center[1]
        
        target_img = cv2.warpAffine(target_img, M, (new_width, new_height), borderValue=(255, 255, 255))
        mask = cv2.warpAffine(mask, M, (new_width, new_height), borderValue=0)

        return target_img, mask

    
    
    
    def add_coco_img_data(self, coco_json, back_img,idx_name):
                    
        height, width, _ = back_img.shape
        
        img_res = {'license': 1, 'file_name': f'{idx_name}.jpg','coco_url': '','height': height,'width': width,'date_captured': '','flickr_url': '','id': idx_name}

        coco_json['images'].append(img_res)
        
        return coco_json
    
    
    
    
    def add_coco_anno_data(self, bbox, idx_name, category_id, coco_json):
        x1 ,y1, x2, y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO bbox 형식으로 변환

        area = coco_bbox[2] * coco_bbox[3]
        res = {'segmentation':[] ,'area': area ,'iscrowd': 0,'image_id': idx_name,'bbox': coco_bbox ,'category_id': category_id,'id': self.id_cnt}
        coco_json['annotations'].append(res)
        
        return coco_json
    
    
    def convert_coordinates(self, x1, y1, x2, y2, img_width, img_height, cate):
        # 바운딩 박스의 중심 좌표와 너비, 높이 계산
        box_width = x2 - x1
        box_height = y2 - y1
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # 이미지의 너비와 높이로 나누어서 정규화된 값 계산
        x_center /= img_width
        y_center /= img_height
        box_width /= img_width
        box_height /= img_height

        # 결과를 문자열로 반환
        return f"{cate} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
    
    def make_yolo_label(self, save_label_path, idx_name, bbox, result_img, category_id):
        height, width = result_img.shape[:2]
        with open(f'{save_label_path}/{idx_name}.txt', "a") as f:
            x1, y1, x2, y2 = bbox[0],bbox[1],bbox[2],bbox[3]
            yolo_bbox = self.convert_coordinates(x1, y1, x2, y2, width, height, category_id)
            f.write(f"{yolo_bbox}\n")
    
    
    def is_bbox_overlap(self, bbox1, bbox2):
        """Check if two bounding boxes overlap."""

        # Extract x, y, width, and height of each bbox
        x1, y1, x1_right, y1_bottom = bbox1
        x2, y2, x2_right, y2_bottom = bbox2

        # Check for overlap
        overlap = (x1 < x2_right) & (x1_right > x2) & (y1 < y2_bottom) & (y1_bottom > y2)

        return overlap
    
    def img_synthesis(self, mask,target_img, back_img,last_bbox):
        try : 
            mask_inv = cv2.bitwise_not(mask)

            height1, width1 = target_img.shape[:2]

            resize_size_w = mask.shape[1]
            resize_size_h = mask.shape[0]

            x_limit = int(back_img.shape[1])-resize_size_w
            y_limit = int(back_img.shape[0])-resize_size_h



            x1 = random.randrange(0,x_limit)
            y1 = random.randrange(0,y_limit)
            x2 = x1 + width1
            y2 = y1 + height1
            bbox = [x1 ,y1, x2, y2]

            flag = True
            for b_bbox in last_bbox : 
                if self.is_bbox_overlap(b_bbox,bbox) == True :
                    flag = False
                    return back_img, None


            roi = back_img[y1:y2, x1:x2]
            fg = cv2.bitwise_and(target_img, target_img, mask=mask_inv)
            bg = cv2.bitwise_and(roi, roi, mask=mask)

            back_img[y1:y2, x1:x2] = cv2.add(fg, bg)
            return back_img, bbox
        
        except Exception as e:
            print(e)
            print(f'back_img : {back_img.shape}')
            print(f'mask : {mask.shape}')
            print(f'resize_size_w : {resize_size_w} resize_size_h : {resize_size_h}')
            print(f'x_limit : {x_limit} y_limit : {y_limit}')
            return back_img, None


    
    
    
    def getAllFilePath(self,root_dir, extensions): 
        img_list = []
        for (root, dirs, files) in os.walk(root_dir):
            if len(files) > 0:
                for file_name in files:
                    if os.path.splitext(file_name)[1] in extensions:
                        img_path = root + '/' + file_name
                        # 경로에서 \를 모두 /로 바꿔줘야함(window버전)
                        img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                        img_list.append(img_path)
        return img_list
    
    
    
    def target_mask_listup(self, rootdir):
        
        dir_list = [it.path for it in os.scandir(rootdir) if it.is_dir() and not it.name.startswith('.')]
        
        target_im_dict = {}
        for idx, target_img_root in enumerate(dir_list) : 
            target_im_files = self.getAllFilePath(target_img_root,[".jpg",".png",".jpeg"])
            category_name = target_img_root.split('/')[-1]
            print(f'category_id : {idx}, category_name : {category_name}')
            target_im_dict[category_name] = {'target_im_files':target_im_files,'category_id':idx }

        return target_im_dict

    
    
    def split_train_dataset(self, img_root_path):

        im_files = self.getAllFilePath(img_root_path,[".jpg",".png",".jpeg"])
        split = int(len(im_files) * self.train_data_r)
        train_img_list = im_files[:split]
        val_img_list = im_files[split:]
        
        return train_img_list, val_img_list

        
    
    
    def make_dataset(self, back_img_list, save_path_list, coco_json, target_im_dict,last_img_cnt):
        
        save_img_path, save_annotations_path = save_path_list[0], save_path_list[1]
        
        for idx, back_img_file in enumerate(back_img_list) : 
            last_bbox = [] 
            back_img = cv2.imread(back_img_file)
            turn = random.randrange(1,4)
            idx_name = idx + last_img_cnt
            coco_json = self.add_coco_img_data(coco_json, back_img,idx_name)
            for i in range(turn) :

                s_lm = int(min(back_img.shape[0],back_img.shape[1])-100)


                category_name = random.choice(list(target_im_dict.keys()))
                category_id = target_im_dict[category_name]['category_id']
                target_im_files = target_im_dict[category_name]['target_im_files']
                target_file = target_im_files[random.randrange(0,len(target_im_files)-1)]
                target_img = cv2.imread(target_file)

                if target_file not in self.mask_dict : 
                    mask = self.remove_background(target_img)
                    self.mask_dict[target_file] = {'mask':mask}
                else :
                    mask = self.mask_dict[target_file]['mask']


                if self.rotation : 
                    target_img, mask = self.rotation_img(target_img, mask)
                    
                    
                if self.resize : 
                    back_img, mask, target_img = self.resize_img(back_img, mask, target_img)
                    
                
                target_img, mask, back_img = self.fit_bbox(target_img, mask, back_img)

                
                result_img, bbox = self.img_synthesis(mask,target_img, back_img,last_bbox)
                if isinstance(bbox, type(None)):
                    continue

                last_bbox.append(bbox)


                if self.yolo_label : 
                    self.make_yolo_label(save_path_list[2], idx_name, bbox, result_img, category_id)


                coco_json = self.add_coco_anno_data(bbox, idx_name, category_id, coco_json)

                self.id_cnt += 1

            if last_bbox : 
                cv2.imwrite(f'{save_img_path}/{idx_name}.jpg',result_img)
                print(f'{idx_name}.jpg')


        with open(save_annotations_path, "w") as f:
            json.dump(coco_json, f)




    def make_json(self,target_im_dict):
        coco_json_train = {}
        coco_json_train['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/','id': 1,
                                        'name': 'Attribution-NonCommercial-ShareAlike License'}]
        coco_json_train['info'] = {'description': 'COCO apple Dataset','url': '','version': '','year': 2023,
                                   'contributor': 'COCO Consortium','date_created': '2023/04/11'}
        coco_json_train['annotations'] = []
        coco_json_train['categories'] = [{'supercategory': cate, 'id': idx, 'name': cate} for idx, cate in enumerate(target_im_dict)]
        coco_json_train['images'] = []


        coco_json_val = {}
        coco_json_val['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/','id': 1,
                                      'name': 'Attribution-NonCommercial-ShareAlike License'}]
        coco_json_val['info'] = {'description': 'COCO apple Dataset','url': '','version': '','year': 2023,
                                 'contributor': 'COCO Consortium','date_created': '2023/04/11'}
        coco_json_val['annotations'] = []
        coco_json_val['categories'] = [{'supercategory': cate, 'id': idx, 'name': cate} for idx, cate in enumerate(target_im_dict)]
        coco_json_val['images'] = []

        return coco_json_train, coco_json_val

    
    def make_dir(self, path):
        if not os.path.exists(path):
            self.make_dir(os.path.dirname(path))
            os.makedirs(path, exist_ok=True)

    
    def make_coco_dir(self, root_path):
        
        if os.path.exists(root_path) : 
            shutil.rmtree(root_path)
        else :
            os.mkdir(root_path)

        save_img_path_train = f'{root_path}/images/train2017'
        save_img_path_val = f'{root_path}/images/val2017'

        save_annotations_path = f'{root_path}/annotations/'
        save_annotations_train = f'{root_path}/annotations/instances_train2017.json'
        save_annotations_val = f'{root_path}/annotations/instances_val2017.json'
        
        if self.yolo_label : 
            save_label_path_train = f'{root_path}/labels/train2017'
            save_label_path_val = f'{root_path}/labels/val2017'

            save_path_list = [save_img_path_train,save_img_path_val,save_label_path_train,
                              save_label_path_val,save_annotations_path]
            
            save_train_path = [save_img_path_train, save_annotations_train, save_label_path_train]
            save_val_path = [save_img_path_val, save_annotations_val, save_label_path_val]
            
        else :
            save_path_list = [save_img_path_train,save_img_path_val,save_annotations_path]
            save_train_path = [save_img_path_train, save_annotations_train]
            save_val_path = [save_img_path_val, save_annotations_val]
        
        
        for save_path in save_path_list : 
            self.make_dir(save_path)
            
        return save_train_path, save_val_path




def main(back_img_path, target_img_path, result_path):
    '''
    back_img_path = '/data/ij/coco_img/val2017'
    target_img_path = '/data/ij/background_remove/target_mask'
    result_path = '/data/ij/background_remove/coco_dataset'
    '''
    
    mcd = Make_Custom_Dataset()
    # mask dataset listup    
    target_im_dict = mcd.target_mask_listup(target_img_path)
    
    #split train dataset & val dataset
    train_img_list, val_img_list = mcd.split_train_dataset(back_img_path)

    # make coco json format
    coco_json_train, coco_json_val = mcd.make_json(target_im_dict)
    
    # make coco dir
    save_train_path, save_val_path = mcd.make_coco_dir(result_path)

    # train
    mcd.make_dataset(train_img_list, save_train_path,coco_json_train,target_im_dict,0)

    # val
    mcd.make_dataset(val_img_list, save_val_path,coco_json_val,target_im_dict,int(len(train_img_list)))
    

    with open(f'{result_path}/category.txt', "a") as f:
        for idx, cate in enumerate(target_im_dict) :
            f.write(f"{idx} : {cate}\n")
    print(f'train_cnt : {len(train_img_list)}, val_cnt : {len(val_img_list)}')
    print('_____________________done_____________________')

    

    

    

    
