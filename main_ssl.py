from training.train_ssl import train_ssl

if __name__ == "__main__":
    best_model_path = train_ssl(
        dataset_path="/home/administrator/Akash/datasets/CSF/",
        epochs=300,
        batch_size=64,
        lr=1e-4,
        save_dir="ssl_checkpoints"
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