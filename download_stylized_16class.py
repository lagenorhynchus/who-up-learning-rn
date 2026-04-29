from modelvshuman import datasets

# This will download the 16-class stylized dataset (approximately 2-3GB)
print("Downloading 16-class Stylized-ImageNet...")
dataset = datasets.StylizedImageNet('/oscar/scratch/3/data/stylized_16class', download=True)
print("Download complete! You can now use this for evaluation.")
