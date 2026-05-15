from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from audioencoder.contrastive import *

def train_alignment_model(
    model: ASTCLIPAlignmentModel, 
    dataloader: DataLoader, 
    epochs: int = 5, 
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    
    # We use AdamW, which is standard for Transformer-based models
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Optional: Freeze the CLIP model if you only want to train the AST side and projection layer
    # for param in model.clip.parameters():
    #     param.requires_grad = False

    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 1. Move data to device
            # Assuming your dataloader yields dictionaries with 'audio_tensors' and 'text_tokens'
            audio_tensors = batch['audio_tensors'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            
            # 2. Forward pass
            audio_embeds, text_embeds, _ = model(
                audio_tensors=audio_tensors, 
                text_tokens=text_tokens
            )
            
            # 3. Compute loss
            loss = model.compute_loss(audio_embeds, text_embeds)
            
            # 4. Backward pass and optimization step
            loss.backward()
            
            # Optional: Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss / len(dataloader):.4f}")

# === Usage Example ===
my_dataset = YourCustomAudioTextDataset(...)
my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, drop_last=True)
model = ASTCLIPAlignmentModel()
train_alignment_model(model, my_dataloader)
