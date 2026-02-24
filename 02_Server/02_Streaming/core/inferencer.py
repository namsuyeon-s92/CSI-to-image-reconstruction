import torch
from models.vae import VAE

class BaseInferencer:
    def __init__(self, model_class, checkpoint_path, device, **kwargs):
        self.device = device
        self.model = model_class.load_from_checkpoint(checkpoint_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def infer(self, input_data):
        input_tensor = torch.from_numpy(input_data).to(self.device)
        with torch.no_grad():
            reconstruction = self.model.decode(self.model.encode(input_tensor))
        return reconstruction
