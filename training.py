from torch.nn import Module
import torchvision
from torchvision import transforms

import wandb
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import UWNetDataSet

from metrics_calculation import *
from model import *
from combined_loss import *

__all__ = [
    "Trainer",
    "setup",
    "training",
]

## TODO: Update config parameter names
## TODO: remove wandb steps
## TODO: Add comments to functions

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader = None):
        device = config['device']
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []

        UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
        wandb.log({f"[Test] Epoch": 0,
                   "[Test] UIQM": np.mean(UIQM),
                   "[Test] SSIM": np.mean(SSIM),
                    "[Test] PSNR": np.mean(PSNR), },
                   commit=True
                   )

        for epoch in trange(0,config.num_epochs,desc = f"[Full Loop]", leave = False):
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0

            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = param_group['lr']*0.7

            for inp, label, _ in tqdm(train_dataloader, desc = f"[Train]", leave = False):
                inp = inp.to(device)
                label = label.to(device)

                self.model.train()

                self.opt.zero_grad()
                out = self.model(inp)
                loss, mse_loss, vgg_loss = self.loss(out, label)

                loss.backward()
                self.opt.step()
                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()

            total_loss_lst.append(total_loss_tmp/len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp/len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp/len(train_dataloader))
            wandb.log({f"[Train] Total Loss" : total_loss_lst[epoch],
                       "[Train] Primary Loss" : primary_loss_lst[epoch],
                       "[Train] VGG Loss" : vgg_loss_lst[epoch],},
                      commit = True
                      )

            if (config.test == True) & (epoch % config.eval_steps == 0):
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                wandb.log({f"[Test] Epoch": epoch+1,
                           "[Test] UIQM" : np.mean(UIQM),
                           "[Test] SSIM" : np.mean(SSIM),
                           "[Test] PSNR" : np.mean(PSNR),},
                          commit = True
                          )

            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, MSE / L1 loss:{}, VGG loss:{}'.format(epoch,config.num_epochs,str(total_loss_lst[epoch]),str(primary_loss_lst[epoch]),str(vgg_loss_lst[epoch])))
                # wandb.log()

            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if epoch % config.snapshot_freq == 0:
                torch.save(self.model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        for i, (img, _, name) in enumerate(test_dataloader):
            with torch.no_grad():
                img = img.to(config.device)
                generate_img = test_model(img)
                torchvision.utils.save_image(generate_img, config.output_images_path + name[0])
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,config.GTr_test_images_path)
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    model = UWnet(num_layers=config.num_layers).to(config["device"])

    transform = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()])
    train_dataset = UWNetDataSet(config.input_images_path,config.label_images_path,transform, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = config.train_batch_size,shuffle = False)
    print("Train Dataset Reading Completed.")

    loss = combinedloss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(model, opt, loss)

    if config.test:
        test_dataset = UWNetDataSet(config.test_images_path, None, transform, False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer
    return train_dataloader, None, model, trainer

def training(config):
    # Logging using wandb
    wandb.init(project = "underwater_image_enhancement_UWNet")
    wandb.config.update(config, allow_val_change = True)
    config = wandb.config

    ds_train, ds_test, model, trainer = setup(config)
    trainer.train(ds_train, config,ds_test)
    print("==================")
    print("Training complete!")
    print("==================")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--input_images_path', type=str, default="./data/input/",help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--label_images_path', type=str, default="./data/label/",help='path of label images(clear images) default:./data/label/')
    parser.add_argument('--test_images_path', type=str, default="./data/input/",help='path of input images(underwater images) for testing default:./data/input/')
    parser.add_argument('--GTr_test_images_path', type = str, default="./data/input/", help='path of input ground truth images(underwater images) for testing default:./data/input/')
    parser.add_argument('--test', default=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step_size',type=int,default=400,help="Period of learning rate decay") #50
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--test_batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
    parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
    parser.add_argument('--print_freq', type=int, default=1)    
    parser.add_argument('--snapshot_freq', type=int, default=2)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    parser.add_argument('--output_images_path', type=str, default="./data/output/")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)

    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    training(config)