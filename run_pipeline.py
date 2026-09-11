import torch
from src.litho_resist.resist_model import ResistUNet

def main():
    print("[*] Starting Lithography Digital Twin Pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device}")
    
    # Initialize and run modules
    model = ResistUNet().to(device)
    dummy_aerial = torch.rand(1, 1, 32, 32).to(device)
    output_silicon = model(dummy_aerial)
    
    print(f"[+] Output shape: {output_silicon.shape}")
    print("[+] Pipeline executed successfully!")

if __name__ == "__main__":
    main()
