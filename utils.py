import torchvision.transforms as T
import matplotlib.pyplot as plt
from pytorch_fid import fid_score
import torch
import os
import torchvision.utils as vutils

def plot_training_loss(G_losses, D_losses, save_path=None):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def generate_images(num_images, netG, latent_dim, device, output_dir='./GENIMG'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    netG.eval()
    # 生成隨機噪聲向量
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    # 使用生成器生成影像
    fake = netG(noise)
    
    # 保存生成的影像
    for j in range(fake.size(0)):
        transform = T.Compose([
            T.Normalize((-1, -1, -1), (2, 2, 2)),
            T.ToPILImage()
        ])
        img = transform(fake[j].cpu())
        img.save(os.path.join(output_dir, 'fake' + str(j) + '.jpg'))
    
    print(f"{num_images} images generated and saved to {output_dir}")

def show_8x8_images(netG, latent_dim, device):
    netG.eval()  # 設置模型為評估模式
    noise = torch.randn(64, latent_dim, 1, 1, device=device)

    with torch.no_grad():  # 禁用梯度計算
        fake_images = netG(noise).detach().cpu()

    grid_image = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=True)

    # 顯示生成的影像
    plt.figure(figsize=(8, 8))  # 調整圖像大小
    plt.axis("off")
    plt.imshow(grid_image.permute(1, 2, 0))  # 將影像維度調整為 (H, W, C)
    plt.show()

def calculate_fid(resized_folder_path, generated_images_folder, batch_size, device, num_workers=0):
    fid_value = fid_score.calculate_fid_given_paths(
        [resized_folder_path, generated_images_folder],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=num_workers
    )
    print('FID value:', fid_value)
    return fid_value
