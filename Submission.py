
from argparse import ArgumentParser
import pandas
import glob
import os
import pydicom
import cv2
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.data.augment import classify_transforms
from ultralytics.yolo.utils.torch_utils import select_device
from pathlib import Path

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'dcm'

class LoadImages:
    # YOLOv8 image/video dataloader, i.e. `yolo predict source=image.jpg/vid.mp4`
    def __init__(self, path, transforms, imgsz=640, stride=32, auto=True, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

            images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
            videos = []
            ni, nv = len(images), len(videos)

            self.imgsz = imgsz
            self.stride = stride
            self.files = images + videos
            self.nf = ni + nv  # number of files
            self.video_flag = [False] * ni + [True] * nv
            self.mode = 'image'
            self.auto = auto
            self.transforms = transforms  # optional
            self.vid_stride = vid_stride  # video frame-rate stride
            self.bs = 1

            if self.nf == 0:
                raise FileNotFoundError(f'No images found in {p}. '
                                        f'Supported formats are:\nimages: {IMG_FORMATS}\n')
    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return self.nf

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        self.count += 1
        im0 = pydicom.dcmread(path).pixel_array
        if im0 is None:
            raise FileNotFoundError(f'Image Not Found {path}')
        s = f'image {self.count}/{self.nf} {path}: '

        im = self.transforms(image=cv2.cvtColor(im0, cv2.COLOR_GRAY2RGB))['image']  # transforms

        return path, im, im0, s


import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.cfg import get_cfg


class ClassificationPredictor:

    def __init__(self, cfg=DEFAULT_CFG, force_one=False, overrides=None, output=None, ):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.output = output
        self.device = select_device(self.args.device)
        self.force_one = force_one

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.device)
        return img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, probs=pred))

        return results

    def setup_model(self, weights, verbose=True):
        from ultralytics.nn.tasks import attempt_load_weights




        model = attempt_load_weights(weights,
                                     device=self.device,
                                     inplace=True)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # model.half() if fp16 else model.float()
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

    def predict_cli(self):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference(self.args.source, self.args.model)

    def write_results(self, idx, results, batch):
        self.output

    def stream_inference(self, source=None, model=None):

        # setup model

        self.setup_model(weights=model)
        # setup source every time predict is called
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        transforms = classify_transforms(self.imgsz[0])
        self.dataset = LoadImages(source,
                             transforms=transforms,
                             imgsz=self.imgsz
                             )
        # check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model

        self.batch = None

        dt = pandas.DataFrame(columns=['path', 'is_patient'])

        if self.force_one:
            self.batch_num = 1
        else:
            self.batch_num = len(self.dataset)
        import time
        t = time.time()
        for cnt, batch in enumerate(self.dataset):

            # self.batch = batch
            path, im, im0s, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

            if cnt % self.batch_num ==0:
                self.path_grp = []
                self.grp = []

            self.grp.append(im)
            self.path_grp.append(path)

            if cnt % self.batch_num == self.batch_num-1:
                ins = torch.stack(self.grp, 0)

            # im = self.preprocess(im)
            # if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                ins = ins.to(self.device)
                preds = self.model(ins)
                # change from here:
                lb_indx_tot = preds.argmax(dim=1).cpu().numpy()
                for path_, lb_indx in zip(self.path_grp, lb_indx_tot):
                    if lb_indx == 0:
                        lb_indx = 1
                    else:
                        lb_indx = 0

                    dt1 = pandas.DataFrame({'path': Path(path_).stem, 'is_patient': [lb_indx]})  # .name changed to .stem

                    dt = pandas.concat([dt, dt1])

        dt.to_csv(self.output, header=False, index=False)
        print("time taken:", time.time()-t)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inputs", type=str, help="path to inference pics")
    parser.add_argument("--output", type=str, help="path to where the csv file is going to be created")
    args = parser.parse_args()

    input = "D:\\IAAA\\datasets\\IAAA\\train\\abnormal"
    output = "j.csv"
    try:
        input = Path(args.inputs)
    except:
        print("sorry the value entered is not some string")

    try:
        output = Path(args.output)
    except:
        print("sorry the value entered is not some string")

    cfg = DEFAULT_CFG
    model = Path(str(__file__)).parent/'ultralytics/yolo/v8/classify/IAAA/126/weights/best.pt'
    source = input
    project = cfg.project or "project_name"
    name = cfg.name or "project_name"
    device = ''  # so the select_device in torch_utils can decide based on the system which to choose, priority on GPU
    args = dict(model=model, source=source, project=project, name=name, device=device)
    predictor = ClassificationPredictor(overrides=args, output=output, force_one= False)
    predictor.predict_cli()
