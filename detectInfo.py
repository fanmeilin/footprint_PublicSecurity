import math
import sys,os
from pathlib import Path
import cv2
from torch import zeros,from_numpy,tensor,no_grad
import numpy as np
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.plots import colors, plot_one_box

from tqdm import tqdm

class Detect_img:
    def __init__(self, weights, device=''):
        # Load model
        set_logging()
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model

    @no_grad()
    def get_info(self,
            source='data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            save_img = False,
            save_dir = "",
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            half=False,  # use FP16 half-precision inference
            ):
        """
        func:利用模型检测图像特征信息
        return:各种特征信息
        """

        # Directories
        os.makedirs(save_dir,exist_ok=True)
        # (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Initialize
        device = self.device
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        model = self.model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        batch_imgbox_list = []  # 存储批量图像的box信息
        batch_small_circle_list = []
        batch_large_circle_list = []
        batch_short_bw_rec_list = []
        batch_short_grey_rec_list = []
        batch_long_bw_rec_list = []
        batch_long_grey_rec_list = []
        img_path_list = []
        for path, img, im0s, vid_cap in tqdm(dataset): #一个batch的信息
            img_path_list.append(path)
            img = from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()

            # Process detections
            # imgbox_list = [] #一张图的box信息
            small_circle_list = []
            large_circle_list = []
            short_bw_rec_list = []
            short_grey_rec_list = []
            long_bw_rec_list = []
            long_grey_rec_list = []

            for i, det in enumerate(pred):  # detections per image 一张图信息
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                p = Path(p)  # to Path
                save_path = save_dir+p.name # img.jpg
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # ["small_circle", "large_circle", "short_bw_rec","short_grey_rec", "long_bw_rec", "long_grey_rec"]
                    # Write results
                    # for *xyxy, conf, cls in reversed(det):
                    for *xyxy, conf, cls in det: #置信度从大到小 #图中的所有框信息
                        xywh_no_norm = xyxy2xywh(tensor(xyxy).view(1, 4)).view(-1).tolist()
                        box = (cls, xywh_no_norm, conf) #存储类别 xywh 置信度
                        if(conf<0.65):continue #过滤置信度小于0.7的框
                        if(cls==0):
                            small_circle_list.append(box)
                        elif(cls==1):
                            large_circle_list.append(box)
                        elif(cls==2):
                            short_bw_rec_list.append(box)
                        elif(cls==3):
                            short_grey_rec_list.append(box)
                        elif(cls==4):
                            long_bw_rec_list.append(box)
                        elif(cls==5):
                            long_grey_rec_list.append(box)
                        # imgbox_list.append(box)
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        # cv2.imwrite(save_path, im0)
                        cv2.imencode('.png', im0)[1].tofile(save_path[:-4]+'.png')  # 英文或中文路径均适用
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # batch_imgbox_list.append(imgbox_list)
            batch_small_circle_list.append(small_circle_list)
            batch_large_circle_list.append(large_circle_list)
            batch_short_bw_rec_list.append(short_bw_rec_list)
            batch_short_grey_rec_list.append(short_grey_rec_list)
            batch_long_bw_rec_list.append(long_bw_rec_list)
            batch_long_grey_rec_list.append(long_grey_rec_list)

        return  img_path_list,(batch_small_circle_list,batch_large_circle_list,batch_short_bw_rec_list,batch_short_grey_rec_list,batch_long_bw_rec_list,batch_long_grey_rec_list)


def adjust_img_list(img_path_list, width, height, save_img, save_dir, padding, batch_small_circle_list,
                    batch_large_circle_list, batch_short_bw_rec_list, batch_short_grey_rec_list, batch_long_bw_rec_list,
                    batch_long_grey_rec_list):
    """
    func:通过模型检测获取的信息，修正足迹 （处理检测出4个圆的图像）
    input:批量小圆列表 批量大圆列表 图像的路径 保存图像的宽高 保存的位置 仿射变换时定点位置的偏移
    return:Rimg_list 修正后的图像列表
    """

    def get_angle(vec1, vec2):
        """
        func: 返回两个向量的夹角（角度制）
        input: vec的形式是[x1,y1,x2,y2]
        return：角度
        """
        dx1 = vec1[2] - vec1[0]
        dy1 = vec1[3] - vec1[1]
        dx2 = vec2[2] - vec2[0]
        dy2 = vec2[3] - vec2[1]
        angle1 = math.atan2(dy1, dx1) * 180 / math.pi  # atan2结果为弧度制(可以处理90度的情况)
        angle2 = math.atan2(dy2, dx2) * 180 / math.pi
        angle = abs(angle1 - angle2)
        if (angle > 180): angle = 360 - angle
        return angle

    def get_dist(xy1, xy2):
        return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)

    def find_circle(small_circle_list, large_circle_list):
        """
        func:寻找用于变换的3个顶点
        input:单图小圆列表 单图大圆列表
        return:坐标轴中心点 长轴顶点 短轴顶点 长短轴比例
        """
        center_xy = [large_circle_list[0][1][0], large_circle_list[0][1][1]]
        dis_list = []
        for box in small_circle_list:
            xy = [box[1][0], box[1][1]]
            dis = math.sqrt((center_xy[0] - xy[0]) ** 2 + (center_xy[1] - xy[1]) ** 2)
            dis_list.append([xy, dis])
        dis_list = sorted(dis_list, key=lambda x: (x[1]), reverse=True)
        ratio = dis_list[0][1] / dis_list[1][1]  # 长轴除以短轴的比例
        return center_xy, dis_list[0][0], dis_list[1][0], ratio

    def find_three_circle(small_circle_list, large_circle_list, angle_thred=30,padding=1):
        same_rular_index = []
        center_xy = [large_circle_list[0][1][0], large_circle_list[0][1][1]]
        # 找到同尺的两个圆索引
        same_rular1, same_rular2 = [],[]
        for i in range(0, 2):
            for j in range(i + 1, 3):
                xy1 = [small_circle_list[i][1][0], small_circle_list[i][1][1]]
                xy2 = [small_circle_list[j][1][0], small_circle_list[j][1][1]]
                vec1 = center_xy + xy1
                vec2 = center_xy + xy2
                if (get_angle(vec1, vec2) < angle_thred):  # 角度阈值
                    same_rular_index = [i, j]
                    same_rular1 = xy1
                    same_rular2 = xy2
        if not(same_rular1 and same_rular2): return [],[]
        other_index = [x for x in [0, 1, 2] if x not in same_rular_index][0]
        other_center = [small_circle_list[other_index][1][0], small_circle_list[other_index][1][1]]
        # 确定长短尺
        same_dist1 = get_dist(center_xy, same_rular1)  # same_dist1为距离短的圆心坐标
        same_dist2 = get_dist(center_xy, same_rular2)
        if (same_dist1 > same_dist2):
            same_dist1, same_dist2 = same_dist2, same_dist1
            same_rular1,same_rular2 = same_rular2,same_rular1
        ratio = same_dist2 / same_dist1
        same_rular = "short_rular"
        if (ratio > 7.5): same_rular = "long_rular"
        same_dist3 = get_dist(center_xy, other_center)
        other_circle = "far"
        if ((same_dist3 / same_dist1) > 0.8 and (same_dist3 / same_dist1) < 1.2): other_circle = "near"
        if (same_rular == "short_rular" and other_circle == "far"):
            scr_xy, dst_xy = [center_xy, other_center, same_rular2], [[padding, padding], [padding, padding+2980], [padding+1580, padding]]
        elif (same_rular == "long_rular" and other_circle == "far"):
            scr_xy, dst_xy = [center_xy, same_rular2, other_center], [[padding, padding], [padding, padding+2980], [padding+1580, padding]]
        elif (same_rular == "short_rular" and other_circle == "near"):
            scr_xy, dst_xy = [center_xy, other_center, same_rular2], [[padding, padding], [padding, padding+280], [padding+1580, padding]]
        else:
            scr_xy, dst_xy = [center_xy, same_rular2, other_center], [[padding, padding], [padding, padding+2980], [padding+280, padding]]
        return scr_xy, dst_xy
    # 针对不同的情况（圆个数）采用不同的解决方案
    def correct(case, i, img,padding=1):
        if not (case in [1, 2, 3]):
            return []
        if (case == 1):
            # assert len(small_circle_list)==4 and len(large_circle_list)==1
            # get points
            center_xy, long_circle_xy, short_circle_xy, ratio = find_circle(small_circle_list, large_circle_list)
            # print("in fact:", ratio)
            # ratio = 1.886
            # revise img
            scr_xy = [center_xy, long_circle_xy, short_circle_xy]
            # dst_xy = [[padding, padding], [padding, padding + ratio * (width - 2 * padding)],
            #           [width - padding, padding]]
            dst_xy = [[padding, padding], [padding, padding+2980], [padding+1580, padding]]

        elif (case == 2):  # 出现三个小圆 一个大圆
            scr_xy, dst_xy = find_three_circle(small_circle_list, large_circle_list,padding=padding)
            if not(scr_xy and dst_xy): return []
        elif (case == 3):  # 出现两个小圆 一个大圆
            short_grey_rec_xy = [short_grey_rec_list[0][1][0], short_grey_rec_list[0][1][1]]
            short_bw_rec_xy = [short_bw_rec_list[0][1][0], short_bw_rec_list[0][1][1]]
            long_grey_rec_xy = [long_grey_rec_list[0][1][0], long_grey_rec_list[0][1][1]]
            scr_xy, dst_xy = [short_grey_rec_xy, short_bw_rec_xy, long_grey_rec_xy], [[padding+500, padding], [padding+980, padding], [padding, padding+2580]]

        scr_xy = np.float32(scr_xy)
        dst_xy = np.float32(dst_xy)
        M = cv2.getAffineTransform(scr_xy, dst_xy)
        Rimg = cv2.warpAffine(img, M, (width, height))
        # Rimg = Rimg[0:height-200,:] #截取图像 剪掉图像的尾部200像素的高 相当于实际上减少了2cm
        return Rimg

    Rimg_list = []
    current_revise = save_dir + "current_revise/"
    # if (save_img and os.path.exists(current_revise)):
    #         rmtree(current_revise)
    for i, img_path in enumerate(tqdm(img_path_list)):
        # img = cv2.imread(img_path)
        img_path_code = np.fromfile(img_path, dtype=np.uint8)  # 含有中文路径时
        img = cv2.imdecode(img_path_code, 1)
        small_circle_list, large_circle_list, short_bw_rec_list, short_grey_rec_list, long_bw_rec_list, long_grey_rec_list = batch_small_circle_list[i], batch_large_circle_list[i], batch_short_bw_rec_list[i], batch_short_grey_rec_list[i], batch_long_bw_rec_list[i], batch_long_grey_rec_list[i]
        # 判断情况的优先级
        case = 0
        if (len(small_circle_list) == 4 and len(large_circle_list) == 1):
            case = 1  # 检测出现5个圆才进行下一步的矫正
        elif (len(small_circle_list) == 3 and len(large_circle_list) == 1):
            case = 2  # 检测出现三个小圆，一个大圆
        elif (len(short_bw_rec_list) == 1 and len(short_grey_rec_list) == 1 and len(long_grey_rec_list)==1):
            case = 3  # 检测处理方块（3个特殊的方块 短尺的灰色方块 短尺的黑白方块 长尺的灰色方块）

        # 分情况处理图像进行矫正
        Rimg = correct(case, i, img,padding=padding)
        p = Path(img_path)
        if(len(Rimg)==0):
            wrong_file = save_dir + 'wrong_file/'
            os.makedirs(wrong_file, exist_ok=True)
            # cv2.imwrite(wrong_file + p.name, img)
            cv2.imencode('.png', img)[1].tofile(wrong_file + p.name[:-4]+'.png')  # 英文或中文路径均适用
            continue
        # print(p.name)
        # save result
        if (save_img):
            gold_img_file = save_dir + "gold_img/"
            revise_result = save_dir + "revise_result/"
            os.makedirs(gold_img_file, exist_ok=True)
            os.makedirs(revise_result, exist_ok=True)
            os.makedirs(current_revise, exist_ok=True)

            # cv2.imwrite(gold_img_file + p.name, img)
            cv2.imencode('.png', img)[1].tofile(gold_img_file + p.name[:-4]+'.png')#英文或中文路径均适用

            # cv2.imwrite(revise_result + p.name, Rimg)
            cv2.imencode('.png', Rimg)[1].tofile(revise_result + p.name[:-4]+'.png')#英文或中文路径均适用

            # cv2.imwrite(current_revise + p.name, Rimg)
            cv2.imencode('.png', Rimg)[1].tofile(current_revise + p.name[:-4]+'.png')#英文或中文路径均适用

        Rimg_list.append(Rimg)

    return Rimg_list


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # libiomp5md.dll,有多个时
    Detect_img = Detect_img(weights="./weights/best.pt")
    img_root = "./data/Saved Pictures/heng8.jpg"
    save_dir_anno = "./data/result/result_temp/annotation/"
    save_dir_img = "./data/result/result_temp/"
    # 获取批量数据的检测信息
    img_path_list,all_info = Detect_img.get_info(source=img_root, save_img=True, save_dir=save_dir_anno)

    padding = 130  # 150->0
    Rimg_list = adjust_img_list(img_path_list, padding+1580+100, padding+2980+200, True, save_dir_img, padding, *all_info) #1810 3310
    # print(batch_imgbox_list)
    # print(batch_small_circle_list)
    # print(batch_large_circle_list)
