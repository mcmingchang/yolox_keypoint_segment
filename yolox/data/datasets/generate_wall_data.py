import numpy as np
import cv2, os, json, shutil, base64, random
from io import BytesIO
from PIL import Image


##### 数据生成
def read_base64(base64_str):
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    return image_data

def base64_2_img(base64_str):
    img_buf = read_base64(base64_str)
    image = Image.open(img_buf)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img


class RandomDataset:
    def __init__(self, path, cate_ls, mask_order, cate_id):
        self.path = path
        self.name_ls, self.json_ls = self.get_data_ls()
        self.cate_ls = cate_ls
        self.mask_order = mask_order
        self.cate_id = cate_id
        self.pre_define_categories = {}
        for i, cls in enumerate(self.cate_ls):
            self.pre_define_categories[cls] = i + 1

    def get_data_ls(self):
        data_ls = os.listdir(self.path)
        name_ls, json_ls = [], []
        for data_name in data_ls:
            name = os.path.splitext(data_name)[-1]
            if name == '.json':
                json_ls.append(data_name)
            elif name in ['.png', '.jpg']:
                name_ls.append(data_name)
            else:
                print(data_name)
        return name_ls, json_ls


    def get_train_data(self):
        if random.randint(0, 1) == 0:
            box_ls = []
            img_name = random.sample(self.name_ls, 1)[0]
            file_name = os.path.splitext(img_name)[0]
            json_name = f'{file_name}.json'
            if json_name in self.json_ls:
                with open(f'{self.path}/{json_name}', 'r') as load_f:
                    load_dict = json.load(load_f)
                img = base64_2_img(load_dict['imageData'])
                h, w, _ = img.shape
                new_w = int((w - 10) / 2)
                left = img[:, :new_w]
                right = img[:, new_w + 10:]
                assert left.shape == right.shape
                for obj in load_dict['shapes']:
                    x_ls, y_ls, seg_ls = [], [], []
                    cate = obj['label']
                    if cate == ['brick wall', 'brick_wall']:
                        cate = 'brick-wall'
                    elif cate in ['wire box', 'wire_box']:
                        cate = 'wire-box'
                    polygon = np.round(np.array(obj['points']))
                    if len(polygon) == 2:
                        x1, y1, x2, y2 = polygon.reshape(-1)
                        polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, new_w)
                    for point in polygon:
                        x, y = point.astype(int)
                        x_ls.append(x)
                        y_ls.append(y)
                        seg_ls.extend([str(x), str(y)])

                    x1, y1, x2, y2 = np.min(x_ls), np.min(y_ls), np.max(x_ls), np.max(y_ls)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    box_ls.append(f'{x1},{y1},{x2},{y2},{self.pre_define_categories[cate]}/{",".join(seg_ls)}')
            else:
                img = Image.open(f'{self.path}/{img_name}')
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                h, w = img.shape[:2]
                new_w = int((w - 10) / 2)
                left = img[:, :new_w]
                right = img[:, new_w + 10:]

            gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            gray = np.expand_dims(gray, axis=-1)
            train_img = np.concatenate([left, gray], axis=-1)
            return train_img, box_ls
        else:
            total_w, data_ls = 0, []
            img_name_ls = random.sample(self.name_ls, 5)
            random.shuffle(img_name_ls)
            for img_name in img_name_ls:
                file_name = os.path.splitext(img_name)[0]
                json_name = f'{file_name}.json'
                mask_dict, polygon_dict = None, None
                if json_name in self.json_ls:
                    with open(f'{self.path}/{json_name}', 'r') as load_f:
                        load_dict = json.load(load_f)
                    img = base64_2_img(load_dict['imageData'])
                    h, w, _ = img.shape
                    new_w = int((w - 10) / 2)
                    if new_w > 400:
                        continue
                    mask_dict = {cate_name: np.zeros((h, new_w)) for cate_name in self.cate_ls}

                    for obj in load_dict['shapes']:
                        cate = obj['label']
                        if cate == ['brick wall', 'brick_wall']:
                            cate = 'brick-wall'
                        elif cate in ['wire box', 'wire_box']:
                            cate = 'wire-box'
                        if cate not in self.cate_ls:
                            print(f'{self.path}/{json_name}', cate)
                        polygon = np.round(np.array(obj['points']))
                        if len(polygon) == 2:
                            x1, y1, x2, y2 = polygon.reshape(-1)
                            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        polygon[:, 0] = np.clip(polygon[:, 0], 0, new_w)
                        polygon = np.expand_dims(polygon, axis=0)
                        cv2.fillPoly(mask_dict[cate], polygon.astype(np.int32), self.cate_id[cate])
                else:
                    img = Image.open(f'{self.path}/{img_name}')
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    h, w = img.shape[:2]
                    new_w = int((w - 10) / 2)
                    if new_w > 400:
                        continue
                total_w += new_w
                left = img[:, :new_w]
                right = img[:, new_w + 10:]
                assert left.shape == right.shape
                gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                gray = np.expand_dims(gray, axis=-1)
                train_img = np.concatenate([left, gray], axis=-1)
                data_ls.append({'img': train_img, 'mask': mask_dict})
            all_img = np.zeros((140, total_w, 4), np.uint8)
            all_mask = np.zeros((140, total_w), np.uint8)
            start_ind = 0
            for data in data_ls:
                img, mask_dict = data['img'], data['mask']
                h, w = img.shape[:2]
                flip = True if random.randint(0, 1) == 0 else False
                if flip:
                    img = cv2.flip(img, 1)
                all_img[:, start_ind:start_ind + w] = img
                if mask_dict is not None:
                    for cate_name in self.mask_order:
                        mask = mask_dict[cate_name]
                        if flip:
                            mask = cv2.flip(mask, 1)
                        all_mask[:, start_ind:start_ind + w][mask > 0] = self.cate_id[cate_name]
                start_ind += w

            ##### 划窗
            h, w = all_img.shape[:2]
            new_w = random.randint(50, min(300 - 1, w))
            start = random.randint(0, w - new_w)

            new_img = all_img[:, start:start + new_w]
            split_mask = all_mask[:, start:start + new_w]
            box_ls = []
            for cate_name in self.mask_order:
                cate_mask = np.zeros((140, new_w), np.uint8)
                cate_mask[split_mask == self.cate_id[cate_name]] = 255
                contours = cv2.findContours(np.array(cate_mask, np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[
                    -2]  # RETR_EXTERNAL  RETR_TREE

                for contour in contours:
                    x_ls, y_ls, seg_ls = [], [], []
                    contour = np.transpose(contour, (1, 0, 2))
                    for item in contour[0]:
                        x, y = item.astype(int)
                        x_ls.append(x)
                        y_ls.append(y)
                        seg_ls.extend([str(x), str(y)])

                    if len(x_ls) > 0:
                        x1, y1, x2, y2 = np.min(x_ls), np.min(y_ls), np.max(x_ls), np.max(y_ls)
                        h_, w_ = y2 - y1, x2 - x1
                        if cate_name in ['door', 'brick-wall'] and w_ < 15:
                            continue
                        if x2 <= x1 or y2 <= y1:
                            continue
                        box_ls.append(f'{x1},{y1},{x2},{y2},{self.pre_define_categories[cate_name]}/{",".join(seg_ls)}')
            return new_img, box_ls


if __name__ == '__main__':
    color = {'door': (0, 0, 255), 'window': (0, 255, 0), 'wire-box': (255, 0, 0),
             'electric-box': (0, 255, 255), 'plaster': (255, 0, 255),
             'brick-wall': (255, 255, 127), 'wallboard': (127, 255, 255),
             'tilt-brick': (255, 127, 255)}
    data_dir, cate_ls, mask_order, cate_id = 'D:/train_model/yolox_keypoint_segment/datasets/wall_seg/dataset', \
                                             ['door', 'window', 'wire-box', 'brick-wall', 'electric-box', 'wallboard', 'tilt-brick'], \
                                             ['brick-wall', 'wallboard', 'tilt-brick', 'door', 'window', 'electric-box', 'wire-box'], \
                                             {'door': 10, 'window': 20, 'wire-box': 30, 'brick-wall': 40, 'electric-box': 50, 'wallboard': 60, 'tilt-brick': 70}
    dataset = RandomDataset(data_dir, cate_ls, mask_order, cate_id)
    for i in range(100):
        img, box_ls = dataset.get_train_data()
        # cv2.imwrite(f'plaster/{i}.png', img)
        left = img[:, :, 0:3].astype(np.uint8)
        right = img[:, :, -1]
        h, w = left.shape[:2]
        mask = np.zeros((h, w, 3))
        for box in box_ls:
            xy, segmentations = box.split("/")
            list_xy = xy.split(",")
            x_min = list_xy[0]
            y_min = list_xy[1]
            x_max = list_xy[2]
            y_max = list_xy[3]
            classes = list_xy[4]
            segmentation = np.array(
                [[int(i) for i in segmentation.split(',')] for segmentation in segmentations.split('*')]).reshape(-1, 2)
            segmentation = np.expand_dims(segmentation, axis=0)
            cv2.fillPoly(mask, segmentation.astype(np.int32), color[dataset.cate_ls[int(classes) - 1]])
        cv2.imshow('1', left)
        cv2.imshow('2', right)

        cv2.imshow('mask', mask)
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()



