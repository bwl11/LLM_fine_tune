import numpy as np
import torch

def test_torch_to_numpy_conversion():
    # Create a sample PyTorch tensor
    torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    
    # Convert the PyTorch tensor to a NumPy array
    numpy_array = torch_tensor.numpy()
    
    # Verify the conversion
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy array failed"
    assert np.array_equal(numpy_array, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)), "Data mismatch after conversion"
    
    print("PyTorch to NumPy conversion test passed.")
