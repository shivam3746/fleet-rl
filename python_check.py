import os, torch, torch_geometric, gymnasium, osmnx
from torch_geometric.nn import GATv2Conv

def main():
    print("CPU cores:", os.cpu_count())
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("torch", torch.__version__, "| PyG", torch_geometric.__version__)

    # Tiny GNN forward pass on GPU
    x = torch.randn(50, 8, device="cuda")
    edge_index = torch.randint(0, 50, (2, 200), device="cuda")
    out = GATv2Conv(8, 16, heads=4).to("cuda")(x, edge_index)
    print("GATv2 output:", out.shape)  # expect [50, 64]

if __name__ == "__main__":
    main()