import torch
import matplotlib.pyplot as plt
import datetime
import os

def plot_mfcc(mfcc_tensor):
    import seaborn as sns
    from IPython.display import display
    """
    Plots MFCC features with shape [time_steps, 29] in a Jupyter notebook.
    
    Parameters:
    - mfcc_tensor: 2D numpy array or tensor of shape [time_steps, 29]
    """
    # Ensure the input is a numpy array or a tensor and convert it to a numpy array if needed
    if isinstance(mfcc_tensor, torch.Tensor):
        mfcc_tensor = mfcc_tensor.numpy()
    
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(mfcc_tensor.T, cmap='rocket_r', cbar=True, annot=False, fmt='.2f')
    plt.title('MFCC Features')
    plt.xlabel('Time Step')
    plt.ylabel('MFCC Coefficient')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

def plot_four_graphs(gt_tensor, reconstructed_tensor, loss, loss_grad, loss_reg,epoch, prefix='', args=None):
    import seaborn as sns
    """
    Plot four graphs: ground truth spectrogram, reconstructed spectrogram, 
    difference between the two, and loss over epoch.
    
    Args:
        gt_tensor (torch.Tensor): Ground truth tensor of shape (batch_size, channels, height, width).
        reconstructed_tensor (torch.Tensor): Reconstructed tensor of same shape as `gt_tensor`.
        loss_array (List[float]): Array of loss values over epochs.
    """
    diff_tensor = torch.abs(gt_tensor - reconstructed_tensor)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Ground Truth Spectrogram
    # use sns to plot this 
    sns.heatmap(gt_tensor.squeeze().T.cpu().numpy(), cmap='rocket_r', ax=axs[0, 0], cbar=True, annot=False, fmt='.2f')
    axs[0, 0].set_title('Ground Truth Spectrogram')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Frequency')

    # Reconstructed Spectrogram
    sns.heatmap(reconstructed_tensor.squeeze().T.cpu().numpy(), cmap='rocket_r', ax=axs[0, 1], cbar=True, annot=False, fmt='.2f')
    axs[0, 1].set_title('Reconstructed Spectrogram')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Frequency')
    
    # Difference Spectrogram
    sns.heatmap(diff_tensor.squeeze().T.cpu().numpy(), cmap='rocket_r', ax=axs[1, 0], cbar=True, annot=False, fmt='.2f')
    # axs[1, 0].imshow(diff_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', aspect='auto')
    axs[1, 0].set_title('Difference Spectrogram')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frequency')

    # Loss over epoch
    axs[1, 1].plot(loss,label='loss')
    axs[1, 1].plot(loss_grad,label='loss gm')
    axs[1, 1].plot(loss_reg,label='loss reg')
    axs[1, 1].set_title('Loss Over Epochs')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    
    plt.tight_layout()
    # save figure with name that has date time hour min and epoch number
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M")
    # save is with os join path args.exp_path
    fig_save_path = os.path.join(args.exp_path, 'figures', '{}{}_{}.png'.format(prefix,now, epoch) ) 
    plt.savefig(fig_save_path)
    plt.show()
