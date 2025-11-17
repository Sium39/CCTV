import torchvision
print(hasattr(torchvision, "_C"))  # Should be True
# If True, try:
print(torchvision._C._has_cuda)    # Should be True
