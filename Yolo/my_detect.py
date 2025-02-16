# -*- coding: utf-8 -*-
import cv2
import torch
import torchvision
import numpy as np
import time
import os
 
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
 
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
 
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
 
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
 
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
 
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
 
 
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
 
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
 
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
 
    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
 
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
 
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
 
        # If none remain process next image
        if not x.shape[0]:
            continue
 
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
 
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
 
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
 
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
 
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
 
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
 
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
 
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
 
    return output
 
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
 
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
 
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def preprocess(img_bgr, img_size, stride, auto):
    # Padded resize
    img_rgb = cv2.resize(img_bgr, img_size)
 
    # Convert
    img_rgb = img_rgb.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_rgb = np.ascontiguousarray(img_rgb)
    return img_rgb, img_bgr
 
class Detect():
    def __init__(self, weights='yolov5s.pt'):
        self.device = 'cpu'
        self.weights =  weights
        self.model = None
        self.imgsz = (640, 640)
        self.conf_thres=0.25
        self.iou_thres=0.45
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = torch.device('cuda:0')
        
        self.init_model()
        self.stride = max(int(self.model.stride.max()), 32)
        
    def init_model(self):
        ckpt = torch.load(self.weights, map_location=self.device)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).float()  # FP32 model
        fuse = True
        self.model = ckpt.fuse().eval() if fuse else ckpt.eval() # fused or un-fused model in eval mode
        self.model.float()

    def draw_boxes_on_image(self, image, pred, conf_thres=0.25, im_size=(640, 640)):
        det = pred[0]
        
        if len(det):
            # 将框从图像大小转换为原图大小
            det[:, :4] = scale_coords(im_size, det[:, :4], image.shape).round()
            
            # 绘制框
            for *xyxy, conf, cls in reversed(det):
                if conf >= conf_thres: 
                    xyxy = [int(coord) for coord in xyxy]
                    
                    # 绘制矩形框，cv2.rectangle参数：图像，左上角坐标，右下角坐标，颜色，线条粗细
                    color = (0, 0, 255)  # 设置矩形框的颜色为绿色
                    image = cv2.rectangle(image, tuple(xyxy[:2]), tuple(xyxy[2:]), color, 2)
                    
                    label = f"Conf {conf:.2f}"
                    cv2.putText(image, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 展示图片
            cv2.imshow("Detection Results", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
    def infer_image(self, image):
        im, im0 = preprocess(image, img_size=self.imgsz, stride=self.stride, auto=True)
        im = torch.from_numpy(im).to(self.device).float() / 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = self.model(im, augment=False, visualize=False)[0]
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)
        # self.draw_boxes_on_image(im0, pred, self.conf_thres)
        return pred
 
 
if __name__ == "__main__":
    detect = Detect('/home/wql/yolov5/best.pt')
    image = cv2.imread('/home/wql/yolov5/detection/images/test/image_182.jpg')
    detect.infer_image(image)
 