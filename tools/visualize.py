import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from io import BytesIO
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

import argparse
import os

# def load(url):
#     """
#     Given an url of an image, downloads the image and
#     returns a PIL image
#     """
#     response = requests.get(url)
#     pil_image = Image.open(BytesIO(response.content)).convert("RGB")
#     # convert to BGR format
#     image = np.array(pil_image)[:, :, [2, 1, 0]]
#     return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )

    dataDir='/depudata1/coco'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds )
    # imgIds = coco.getImgIds(imgIds = [341681])
    curr_img_id = imgIds[np.random.randint(0,len(imgIds))]
    img = coco.loadImgs(curr_img_id)[0]

    image = io.imread(img['coco_url'])
    # image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
    plt.imshow(image)
    predictions = coco_demo.run_on_opencv_image(image)
    plt.imshow(predictions)

    str1 = '%s.jpg' % curr_img_id
    plt.savefig(str1)

if __name__ == "__main__":
    main()