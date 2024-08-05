import torch

def inspect_pt_file(file_path):
    # Load the .pt file
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    
    # Print the keys in the checkpoint
    print("Keys in the checkpoint:")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    # Print the model structure
    if 'model' in checkpoint:
        model = checkpoint['model']
        print("\nModel structure:")
        print(model)
    
    # Print optimizer state if available
    if 'optimizer' in checkpoint:
        optimizer = checkpoint['optimizer']
        print("\nOptimizer state dict:")
        print(optimizer)
    
    # Print other available information
    if 'epoch' in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    if 'best_fitness' in checkpoint:
        print(f"Best fitness: {checkpoint['best_fitness']}")
    if 'training_results' in checkpoint:
        print(f"Training results: {checkpoint['training_results']}")

# Example usage
file_path = 'D:/yolov5-face-mask-detection-master/face-mask-detection/models/mask_yolov5.pt'
inspect_pt_file(file_path)
