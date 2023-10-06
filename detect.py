
import torch.nn as nn

import torch
import numpy as np
from torchvision import transforms
import time
from PIL import Image, ImageDraw
import cv2
from imutils.video import WebcamVideoStream
import os
import tensorflow as tf
from torchvision.ops.boxes import batched_nms, nms
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#a = cv2.videocapture
device = torch.device('cpu')
def convert_to_square(boxes):
        if boxes.shape[0] == 0:
            return np.array([])
        square_boxes = boxes.copy()
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        max_side = np.maximum(w, h)
        square_boxes[:, 0] = boxes[:, 0] + w / 2 - max_side / 2
        square_boxes[:, 1] = boxes[:, 1] + h / 2 - max_side / 2
        square_boxes[:, 2] = square_boxes[:, 0] + max_side
        square_boxes[:, 3] = square_boxes[:, 1] + max_side
        return square_boxes

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1),
            nn.BatchNorm2d(10),
            nn.PReLU(10),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32)
        )

        self.conv4_1 = nn.Conv2d(32, 1, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)
    def forward(self, x):
        x = self.pre_layer(x)
        cls = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cls, offset
    
class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1),
            nn.BatchNorm2d(28),
            nn.PReLU(28),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(28, 48, 3, 1),
            nn.BatchNorm2d(48),
            nn.PReLU(48),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU(128)
        )
        self.linear5_1 = nn.Linear(128, 1)
        self.linear5_2 = nn.Linear(128, 4)
    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear4(x)
        cls = torch.sigmoid(self.linear5_1(x))
        offset = self.linear5_2(x)
        return cls, offset
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),  # 46
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, padding=1),  # 23
            nn.Conv2d(32, 64, 3, 1),  # 21
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2),  # 10
            nn.Conv2d(64, 64, 3, 1),  # 8
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2),  # 4
            nn.Conv2d(64, 128, 2, 1),  # 3
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )
        self.linear5 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(256)
        )
        self.linear6_1 = nn.Linear(256, 1)
        self.linear6_2 = nn.Linear(256, 4)
        self.linear6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(-1, 128*3*3)
        x = self.linear5(x)
        cls = torch.sigmoid(self.linear6_1(x))
        offset = self.linear6_2(x)
        point = self.linear6_3(x)
        return cls, offset, point
    
'''
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),  # 46
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, padding=1),  # 23
            nn.Conv2d(32, 64, 3, 1),  # 21
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2),  # 10
            nn.Conv2d(64, 64, 3, 1),  # 8
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2),  # 4
            nn.Conv2d(64, 128, 2, 1),  # 3
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )
        self.linear5 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(256)
        )
        self.linear6_1 = nn.Linear(256, 1)
        self.linear6_2 = nn.Linear(256, 4)
        self.linear6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(-1, 128*3*3)
        x = self.linear5(x)
        cls = torch.sigmoid(self.linear6_1(x))
        offset = self.linear6_2(x)
        point = self.linear6_3(x)
        return cls, offset, point
'''
def iou(box, boxes, isMin=False):
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 交集的左上角取两个框大的值，右下角取两个款小的值
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    # 求交集的边长，最短为0
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    inter = w * h

    if isMin:
        return np.divide(inter, np.minimum(area, areas))
    else:
        return np.divide(inter, area + areas - inter)

cv2.resize

def Nms(boxes, thresh, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    # 将框根据置信度从大到小排序
    _boxes = boxes[(-boxes[:]).argsort()]
    r_box = []
    # 计算iou，留下iou值小的框
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]
        r_box.append(a_box)
        index = np.where(iou(a_box, b_boxes, isMin) < thresh)
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_box.append(_boxes[0])

    return np.stack(r_box)



class Detector():
    def __init__(self, pnet_path,rnet_path,onet_path,softnms=False, thresholds=None, factor=0.707):
        if thresholds is None:
            thresholds = [0.7, 0.8, 0.9]
        self.thresholds = thresholds
        self.factor = factor
        self.softnms = softnms

        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.load_state_dict(torch.load(pnet_path,device))
        self.rnet.load_state_dict(torch.load(rnet_path,device))
        self.onet.load_state_dict(torch.load(onet_path,device))
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        

        self.img_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    

    def detect(self, image):
            
        starttime = time.time()
        pnet_boxes = self.pnet_detect(image)
        #if pnet_boxes.shape[0] == 0:
            #print("P网络为检测到人脸")
            #return np.array([])
        
       
        #print(pnet_time)
        
        
       
        
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        #if rnet_boxes.shape[0] == 0:
            #print("R网络为检测到人脸")
        endtime = time.time()
        fps = 1/(endtime-starttime)
        onet_boxes = self.onet_detect(image, rnet_boxes)
        return pnet_boxes,rnet_boxes,onet_boxes,fps
       
        

      
        

       
      
    
    
    def pnet_detect(self, image):
        boxes = []
        w = image.shape[1]
        h = image.shape[0]
        min_side = min(w, h)
        scale = 1
        i = 0
        #去除第一张
        # scale = 0.7
        # image = image.resize((int(w*scale), int(h*scale)))
        while min_side > 12:
            img_data = self.img_transfrom(image).to(device)
            img_data.unsqueeze_(0)
            _cls, _offset = self.pnet(img_data)
            #print(_cls)
            _cls = _cls[0][0].data.cpu()
            _offset = _offset[0].data.cpu()
            print(_cls.shape)
            # (n,2)
            indexes = torch.nonzero(_cls > self.thresholds[0])
            
            #print(indexes)
            # for循环改进
            # for index in indexes:
            #     boxes.append(self.box(index, _cls[index[0], index[1]], _offset, scale))
            boxes.extend(self.box(indexes, _cls, _offset, scale))
            
            scale *= self.factor
            _w = int(w * scale)
            _h = int(h * scale)
            image = cv2.resize(image,(_w, _h))
            min_side = min(_w, _h)
            i = i+1
        #print(i)    
        #boxes = np.array(boxes)
        #print(boxes[:])
        if (len(boxes)) >0 :
            boxes = torch.stack(boxes)
            return boxes[nms(boxes[:,:4],boxes[:,4] ,0.7)].numpy()
        else:
            return np.array([])
            
        
       
    
    
    
    def rnet_detect(self, image, pnet_boxes):
        boxes = []
        img_dataset = []
        # 取出正方形框并转成tensor，方便后面用tensor去索引
        square_boxes = torch.from_numpy((pnet_boxes))
        for box in square_boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            # crop裁剪的时候超出原图大小的坐标会自动填充为黑色
            #print(_x1,_y1,_x2,_y2)
            
            if (_x1<0):
                _x1=0
            if (_y1<0):
                _y1=0
            if (_x2>image.shape[1]):
                _x2=image.shape[1]
            if (_y2>image.shape[0]):
                _y2=image.shape[0]
            
        
            img_crop = image[_y1:_y2,_x1:_x2]
            img_crop1 = cv2.resize(img_crop,(24,24))
           
            img_data = self.img_transfrom(img_crop1).to(device)
            img_dataset.append(img_data)

        if  (len(img_dataset))>0:
            _cls, _offset = self.rnet(torch.stack(img_dataset))
        #print(_offset)
            _cls = _cls.data.cpu()
            _offset = _offset.data.cpu()

        # (14,)
            indexes = torch.nonzero(_cls > self.thresholds[1])[:,0]
            #print(indexes)
        # (n,5)
        

            box = square_boxes[indexes]
        #print(box)
        # (n,)
            #print((box.ndim))
            if ((box.ndim) == 1):
                _x1 = box[0]
                _y1 = box[1]
                _x2 = box[2]
                _y2 = box[3]
            else:
                _x1 = box[:, 0]
                _y1 = box[:, 1]
                _x2 = box[:, 2]
                _y2 = box[:, 3]

            sidew = _x2 - _x1
            sideh  = _y2-_y1

        # (n,4)
            offset = _offset[indexes]
        # (n,)
            x1 = _x1 + sidew * offset[:, 0]
            y1 = _y1 + sideh * offset[:, 1]
            x2 = _x2 + sidew * offset[:, 2]
            y2 = _y2 + sideh * offset[:, 3]
        # (n,)
            #print(_cls)
            cls = _cls[indexes][:,0]
            #print(cls)
        # np.array([x1, y1, x2, y2, cls]) (5,n)
            boxes.extend(torch.stack([x1, y1, x2, y2, cls], dim=1))
            if len(boxes) >0:
                boxes = torch.stack(boxes)
                return boxes[nms(boxes[:, :4], boxes[:, 4], 0.7)].numpy()
            else:
                return np.array([])
        # (n,1) (n,4)
        else :
            return np.array([])
       
        
        


            
        #print(boxes[nms(boxes[:, :4], boxes[:, 4], 0.9)].numpy())

             
    
    def onet_detect(self, image, rnet_boxes):
        boxes = []
        img_dataset = []
        square_boxes = torch.from_numpy((rnet_boxes))
        for box in square_boxes:
           

            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            if (_x1<0):
                _x1=0
            if (_y1<0):
                _y1=0
            if (_x2>image.shape[1]):
                _x2=image.shape[1]
            if (_y2>image.shape[0]):
                _y2=image.shape[0]
            img_crop = image[_y1:_y2,_x1:_x2]
            
            img_crop = cv2.resize(img_crop,(48, 48))
            img_data = self.img_transfrom(img_crop).to(device)
            img_dataset.append(img_data)

        if (len(img_dataset)) > 0:
                _cls, _offset, _point = self.onet(torch.stack(img_dataset))
                _cls = _cls.data.cpu()
                _offset = _offset.data.cpu()
                _point = _point.data.cpu()
                indexes, _ = np.where(_cls > self.thresholds[2])
        # (n,5)
                box = square_boxes[indexes]
        # (n,)
                _x1 = box[:,0]
                _y1 = box[:,1]
                _x2 = box[:,2]
                _y2 = box[:,3]
                side = _x2 - _x1
        # (n,4)
                offset = _offset[indexes]
        # (n,)
                x1 = _x1 + side * offset[:, 0]
                y1 = _y1 + side * offset[:, 1]
                x2 = _x2 + side * offset[:, 2]
                y2 = _y2 + side * offset[:, 3]
        # (n,)
                cls = _cls[indexes][:, 0]
        # (n,10)
                #print(cls)
                point = _point[indexes]
                px1 = _x1 + side * point[:, 0]
                py1 = _y1 + side * point[:, 1]
                px2 = _x1 + side * point[:, 2]
                py2 = _y1 + side * point[:, 3]
                px3 = _x1 + side * point[:, 4]
                py3 = _y1 + side * point[:, 5]
                px4 = _x1 + side * point[:, 6]
                py4 = _y1 + side * point[:, 7]
                px5 = _x1 + side * point[:, 8]
                py5 = _y1 + side * point[:, 9]
        # np.array([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5]) (15,n)
                boxes.extend(torch.stack([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5], dim=1))

                
                if len(boxes)>0:
                    boxes = torch.stack(boxes)
                    return boxes[nms(boxes[:, :4], boxes[:, 4], 0.2)].numpy()
                else:
                    return np.array([])
               
                
        else :
            return np.array([])

        
        
    
    def box(self, indexes, cls, offset, scale, stride=2, side_len=12):
        # (n,)
        _x1 = (indexes[:, 1] * stride) / scale
        _y1 = (indexes[:, 0] * stride) / scale
        _x2 = (indexes[:, 1] * stride + side_len) / scale
        _y2 = (indexes[:, 0] * stride + side_len) / scale
        side = _x2 - _x1
        # (4, n)
        offset = offset[:, indexes[:, 0], indexes[:, 1]]
        # (n,)
        x1 = (_x1 + side * offset[0])
        y1 = (_y1 + side * offset[1])
        x2 = (_x2 + side* offset[2])
        y2 = (_y2 + side* offset[3])

        # (n,)
        cls = cls[indexes[:, 0], indexes[:, 1]]

        # (n, 5)
        return torch.stack([x1, y1, x2, y2,cls], dim=1)
'''
        start_time = time.time()
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            print("R网络为检测到人脸")
            return np.array([])
        end_time = time.time()
        rnet_time = end_time - start_time

      

        start_time = time.time()
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            print("R网络为检测到人脸")
            return np.array([])
        end_time = time.time()
        rnet_time = end_time - start_time

        start_time = time.time()
        onet_boxes = self.onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            print("O网路未检测到人脸")
            return np.array([])
        end_time = time.time()
        onet_time = end_time - start_time

        sum_time = pnet_time + rnet_time + onet_time
        print("time:{}, pnet_time:{}, rnet_time:{}, onet_time:{}".format(sum_time, pnet_time, rnet_time, onet_time))
'''
   
'''
    ret, img= cv2.threshold(img, 127, 255, 0)
# 尋找輪廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    print(len(contours[0]))
    cv2.drawContours(img, contours, -1, (0,255,0), 3)

    img = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)

    #img1 = cv2.imread(img_path)
    #img1 = cv2.resize(img1,(200,200))

'''  
''''
for (data,dec) in zip(prediction,decs):
                (with_mask,without_mask)= data
                (startX,starY,endX,endY) = dec
                if (with_mask>without_mask):
            #text('good')
                    cv2.rectangle(img,(startX,starY),(endX,endY),(0,255,0))
                elif (with_mask<without_mask):
            #text('bad')
                    cv2.rectangle(img,(startX,starY),(endX,endY),(0,0,255))
'''

if __name__ == '__main__':
    model =tf.keras.models.load_model("maskdetector.model", compile=False)
    faces_array= np.ndarray(shape=(1, 224,224, 3), dtype=np.float32)
    img_path = 'D:\exam\human/maskface13.jpg'
    cap = cv2.VideoCapture(0) 
    #cap.set(cv2.CAP_PROP_FPS,10)
    #fps =  cap.get(cv2.CAP_PROP_FPS)
    fps2 =  cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 取得畫面尺寸
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
# 使用 XVID 編碼
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output127.avi',fourcc, 10, (width,height),True)
# 建立 VideoWriter 物件，輸出影片至 output.avi，FPS 值為 20.0
    

    
    
    
    #if (width<200):
    #    width = 200
    #if (height<200):
    #    height = 200
    '''
    a = [[1,2,3],[4,5,6]]
    print(a[1])
    b = a
    b =np.array(a)
    print(b)
    prit(b[:,0])
    '''

        
        #frame = [1]*10000
    a =0
    #for i in range(-10,10):
        #for j in range (-10,10):
            #img = cv2.imread(img_path)
            #height = img.shape[0]
            #width =  img.shape[1]
            #print(height)
            #print(width)
            #print(img.shape)

            
            #img = cv2.resize(img,(100,100))
            

            #print(img.shape)
    #img = img[int(0.2*img.shape[0]):int(0.7*img.shape[0]),int(0.3*img.shape[1]):int(0.8*img.shape[1])]
    while (True):
        clsarray = []
        clsarray2 =[]
        clsarray3 = []
        decs2 = []
        decs = []
        decs1 = []
        faces2 = []
        faces_array = []
        faces = []
        prediction = []
        #frame = cv2.imread(img_path)
        ret,frame = cap.read()
        
        
        #frame1 = frame
        #frame1= cv2.resize(frame1,(500,500))
            #frame1 = frame.copyTo()
        #frame = cv2.resize(frame,(200,200))
        #frame = cv2.cvtColor(frame, )
        
           
            


            #if (len(onet_boxes)) == 0:
                #continue
    
        starttime = time.time()
        print(starttime)
        #frame = cv2.resize(frame,(300,280))
        
        detector = Detector("p_net-25.pth","r_net-12.pth",'o_net-3.pth')
        pnet_boxes,rnet_boxes,onet_boxes,fps1 = detector.detect(frame)
        
        pnet_boxes = pnet_boxes.tolist()
        rnet_boxes = rnet_boxes.tolist()
        onet_boxes = onet_boxes.tolist()
        for (k,position) in enumerate(onet_boxes):
                x = int(position [0])
                y = int(position [1])
                w = int(position [2])
                h = int(position [3])
                cls = position[4]
                #clsarray.append(cls)

                clsarray3.append(cls)
                
                print((x,y,w,h,cls))
                
                    

                face = frame[int(y):int(h),int(x):int(w)]
                face = cv2.resize(frame,(224,224))
                #face = img_to_array(face)
                #face = preprocess_input(face)
                faces.append(face)
                decs.append((x,y,w,h))
                decs2.append((x,y,w,h))
        #cv2.rectangle(img1,(int(x),int(y)),(int(w),int(h)),(0,255,0))
        if (len(rnet_boxes)>1):
                clsarray5 =  clsarray3
                #clsarray5 =  torch.tensor(clsarray5)
                clsarray3 = []
                decs3 = decs2
                decs1.append((decs3))
                decs2 = []
                #for i in range(0,len(onet_boxes)):
                clsarray2.append(clsarray5)

        index = []
        pointlist = []
            #clsarray = np.array(clsarray)
        faces2= np.array(faces)
        faces_array = (faces2.astype(np.float32) / 127.0) - 1
        if (len(faces_array)>0):
            prediction = model.predict(faces_array)
        else:
            prediction = []

        for (data,point) in zip(prediction,decs):
            (with_mask1,without_mask1) = data
            if (with_mask1>without_mask1):
            #text('good')
                    startX1 = point[0]
                    startY1 =  point[1]
                    endX1 = point[2]
                    endY1 = point[3]
                    cv2.rectangle(frame,(startX1,startY1),(endX1,endY1),(0,255,0))
            elif (with_mask1<without_mask1):
            #text('bad')
                    startX1 = point[0]
                    startY1 =  point[1]
                    endX1 = point[2]
                    endY1 = point[3]
                    cv2.rectangle(frame,(startX1,startY1),(endX1,endY1),(0,0,255))
            
        endtime = time.time()
        print(endtime)
        fps = (1/(endtime-starttime))
        cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
        out.write(frame)
        a = 1
            
        cv2.imshow('aaa',frame)
        
        #cv2.imwrite(str(a),frame1)
        
        #time.sleep(2.0)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            #break
            cap.release()
            cv2.destroyAllWindows()
            #out.release()
            break
    
        
            #frame = cv2.resize(frame,(1,1))
    
        #if cv2.waitKey(0) :
            #cv2.destroyAllWindows()
    
            
            #cv2.destroyAllWindows()
            #if cv2.waitKey(0) :
               # break
                #cv2.destroyAllWindows()
    #array2 = np.array(clsarray2)
    
    #print(array2)
    '''
    clsarray1 = []
    thre = []
    #k = 0
    j = 0
    k =0
    e = 0
    if (len(onet_boxes)>1):
        for i in range(0,len(decs1)):
            j = 0
            k =0
            while (k < len(decs1[i])):
                #print(j)
                if((decs1[i][k][0])>(decs1[i][j][0])):
                    a = decs1[i][k]
                    decs1[i][k] = decs1[i][j]
                    decs1[i][j] = a
                
                if (j >= (len(decs1[i])-1)):
                    k = k+1
                    j = k
                elif (j<(len(decs1[i])-1)):
                    j = j+1
                '''
    

            
            
             
           

            
               
    #print(decs1)

    '''
    if (len(onet_boxes)<=1):
        index =  [np.argsort(-clsarray)]
        index1 = index[0][0]
    #index = index.tolist()
    #index = index[0]
        #decs = torch.tensor(np.array(decs))
        print(decs)
        print(index)
        (startX) = decs[index1][0]
        starY = decs[index1][1]
        endX =decs[index1][2]
        endY = decs[index1][3]
        (with_mask)= prediction[index1,0]
        without_mask=prediction[index1,1]
        thre.append((with_mask,without_mask))
        pointlist.append((startX,starY,endX,endY))

    else:
        a = 0
        
        average = []
        score = 0
        for i in range((len(clsarray2))):
            score = 0
            for j in range((len(clsarray2[i]))):
                score = score+clsarray2[i][j]

            
            average.append((score/len(clsarray2[i])))

        average = np.array(average)
        index = [np.argsort(-average)]
        ind = index[0][0]
        #index =  [np.argsort(clsarray2,axis=0)]
        #ind = index[-1][a]

        #ind1 = index[-1]
        #clsarray2 = clsarray2.tolist()
        while(a<len(decs1[ind])):
            clsarray4 = []
            
            #index = np.where(clsarray2==clsarray4[0])
            
            #print(clsarray4)
            #print(np.array(decs1))
            #print(index)
            
            #print(ind)
            
            #index = np.array(index)
            #index = index.tolist()
            print(index)
            print(len(index))
        
            #for i in range(0,len(index)):
                #ind = index[-1-i][a]
                
                #print(len(decs1[ind]))
               # print(a)
                #if (a>(len(decs1[ind])-1)):
                 #   continue
                #else:
                    #break
            
            print(ind)
            print(decs1[ind])
            print(clsarray2[ind])
            #decs1 = torch.from_numpy(np.array(decs1))
            startX= decs1[ind][a][0]
            starY = decs1[ind][a][1]
            endX =decs1[ind][a][2]
            endY = decs1[ind][a][3]
            
            

            (with_mask) = prediction[ind,0]
            without_mask=prediction[ind,1]
            thre.append((with_mask,without_mask))
            pointlist.append((startX,starY,endX,endY))
            a = a+1

            
        
        
        
    #print(startX)
    print(pointlist)
    for (data,point) in zip(thre,pointlist):
        (with_mask1,without_mask1) = data
        if (with_mask1>without_mask1):
            #text('good')
            startX1 = point[0]
            startY1 =  point[1]
            endX1 = point[2]
            endY1 = point[3]
            cv2.rectangle(img,(startX1,startY1),(endX1,endY1),(0,255,0))
        elif (with_mask1<without_mask1):
            #text('bad')
            startX1 = point[0]
            startY1 =  point[1]
            endX1 = point[2]
            endY1 = point[3]
            cv2.rectangle(img,(startX1,startY1),(endX1,endY1),(0,0,255))

    cv2.imshow('aaa',img)
    if cv2.waitKey(0) :
        cv2.destroyAllWindows()
            '''

        
