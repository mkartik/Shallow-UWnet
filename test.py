import torch
import numpy as np
import torchvision
from torchvision import transforms

import argparse
import time
from tqdm import tqdm

from model import *
from dataloader import UWNetDataSet
from metrics_calculation import *

__all__ = [
    "test",
    "setup",
    "testing",
]

@torch.no_grad()
def test(config, test_dataloader, test_model):
    test_model.eval()
    for i, (img, _, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(config.device)
            generate_img = test_model(img)
            torchvision.utils.save_image(generate_img, config.output_images_path + name[0])

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    model = torch.load(config.snapshot_path).to(config.device)

    transform = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()])
    test_dataset = UWNetDataSet(config.test_images_path,None,transform, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = config.batch_size,shuffle = False)
    print("Test Dataset Reading Completed.")
    return test_dataloader, model

def testing(config):
    ds_test, model = setup(config)
    test(config, ds_test, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot_path',type=str,default='./snapshots/model.ckpt',help='snapshot path,such as :xxx/snapshots/model.ckpt default:None')
    parser.add_argument('--test_images_path', type=str, default="./data/input/",help='path of input images(underwater images) for testing default:./data/input/')
    parser.add_argument('--output_images_path',type=str,default='./data/output/',help='path to save generated image.')
    parser.add_argument('--batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
    
    parser.add_argument('--calculate_metrics', type=bool, default=False, help="calculate PSNR, SSIM and UIQM on test images")
    parser.add_argument('--label_images_path', type=str, default="./data/label/",help='path of label images(clear images) default:./data/label/')


    print("-------------------testing---------------------")
    config = parser.parse_args()
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)

    start_time = time.time()
    testing(config)
    print("total testing time" , time.time() - start_time)

    if config.calculate_metrics:
        print("-------------------calculating performance metrics---------------------")
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path, config.label_images_path, (config.resize, config.resize))
        UIQM_measures = calculate_UIQM(config.output_images_path, (config.resize, config.resize))

        print("SSIM on {0} samples {1} ± {2}".format(len(SSIM_measures), np.round(np.mean(SSIM_measures), 3), np.round(np.std(SSIM_measures), 3)))
        print("PSNR on {0} samples {1} ± {2}".format(len(PSNR_measures), np.round(np.mean(PSNR_measures), 3), np.round(np.std(PSNR_measures), 3)))
        print("UIQM on {0} samples {1} ± {2}".format(len(UIQM_measures), np.round(np.mean(UIQM_measures), 3), np.round(np.std(UIQM_measures), 3)))
