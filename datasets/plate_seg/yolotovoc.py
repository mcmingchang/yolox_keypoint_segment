from lxml.etree import Element, SubElement, tostring
import cv2, os

CATE_LS = []

class XmlMaker:

    def __init__(self,txtpath,xmlpath):
        self.txtPath = txtpath
        self.xmlPath = xmlpath
        self.txtList = []
        self.keypoint = False
        self.segmentation = True

    def readtxt(self):
        txtfile = open(self.txtPath, "r", encoding='gbk', errors='ignore')
        self.txtList = txtfile.readlines()
        for i in self.txtList:
            jpg = i.strip().split(" ")[0]
            if self.keypoint:
                keypoints = i.strip().split(" ")[1]
                xys = i.strip().split(" ")[2:]
            elif self.segmentation:
                xys = i.strip().split(" ")[1:]
            else:
                xys = i.strip().split(" ")[1:]

            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'VOC2012'
            node_filename = SubElement(node_root, 'filename')
            node_filename.text = jpg
            img = cv2.imread(f'images/{jpg}')
            shape = img.shape

            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')

            node_width.text = str(shape[1])

            node_height = SubElement(node_size, 'height')
            node_height.text = str(shape[0])

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = '3'

            for xy in xys:
                if self.segmentation:
                    xy, segmentations = xy.split("/seg/")
                    list_xy = xy.split(",")
                    x_min = list_xy[0]
                    y_min = list_xy[1]
                    x_max = list_xy[2]
                    y_max = list_xy[3]
                    classes = list_xy[4]
                else:
                    list_xy = xy.split(",")
                    x_min = list_xy[0]
                    y_min = list_xy[1]
                    x_max = list_xy[2]
                    y_max = list_xy[3]
                    classes = list_xy[4]
                CATE_LS.append(classes)
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(x_min)
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(y_min)
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(x_max)
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(y_max)

                if self.keypoint:
                    node_keypint = SubElement(node_object, 'keypoints')
                    node_keypint.text = keypoints
                elif self.segmentation:
                    node_segmentation = SubElement(node_object, 'segmentation')
                    node_segmentation.text = segmentations


            xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
            xml_name = jpg.split("/")[-1][:-4]+".xml"
            print(xml_name)
            with open(self.xmlPath+"/"+xml_name, "wb") as f:
                f.write(xml)
                f.close()

if __name__ == "__main__":
    read =XmlMaker("train.txt","images")
    read.readtxt()
    os.rename('./images', './nnotations')
    print(set(CATE_LS))

