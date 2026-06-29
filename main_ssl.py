from training.train_ssl import train_ssl

if __name__ == "__main__":
    experiment_id = "Experiment7"
    best_model_path = train_ssl(
        #dataset_path="/work/scratch/ak19jybi/datasets/zenodo/",
	dataset_path="/work/scratch/ak19jybi/datasets/bloodsmear/",
        epochs=300,
        batch_size=64,
        lr=1e-4,
        save_dir=f"/work/scratch/ak19jybi/project/ssl/dino/ssl_checkpoints/{experiment_id}/",
        resume=True,
        checkpoint_path=f"/work/scratch/ak19jybi/project/ssl/dino/ssl_checkpoints/{experiment_id}/checkpoint_epoch_75.pt",
        log_dir=f"/work/scratch/ak19jybi/project/ssl/dino/ssl_tensorboard/{experiment_id}/"
    )

    print("Best model saved at:", best_model_path)

    '''
    from utils.viz import visualize_embeddings
    from datasets.ssl_dataset import SSLImageFolder
    from torch.utils.data import DataLoader
    import torch

    # load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval().cuda()

    # build small sample loader
    dataset = SSLImageFolder("data/csf_images") # load csf500
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    visualize_embeddings(model, loader, device="cuda", save_path="ssl_checkpoints/ssl_tsne.png")
    '''
