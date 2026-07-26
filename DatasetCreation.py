from prepare_dataset import dataloaders
from get_args import args
import torch
import os

#Parameter
batch_size_train = 64
InputSize = 5
Downscale_batch_size = 100_000

args.selected_targets = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
args.batch_size = batch_size_train

#Options
args.shuffle = True
args.downscale = True
args.flatten_dataset = True
args.dataset_spin_form = False
args.pad_flattened_dataset = True  # must flatten_dataset be True to pad the dataset
args.remove_contradicting = True

#Adaptive
args.use_adaptive = True

args.downscale_batch_size = Downscale_batch_size
args.adaptive_avg_pool_shape = InputSize  # 5-> (5x5), 6-> (6x6)
if args.use_adaptive and args.downscale:
    args.shape_2d = (args.adaptive_avg_pool_shape, args.adaptive_avg_pool_shape) 


#Overwrite protection
# The repository ships a dataset for each image size, and every Info.txt and
# every QUBO was generated from those exact files. Regenerating the dataset
# shuffles it and can therefore change the verified input, the perturbable
# pixels and the QUBO, which silently invalidates the shipped results. So an
# existing dataset is never overwritten unless it is asked for explicitly:
#
#     OVERWRITE=1 python DatasetCreation.py
#
Overwrite = os.environ.get("OVERWRITE", "0").lower() in ("1", "true", "yes")

Folder = f"Dataset/{InputSize}x{InputSize}"
TrainFile = Folder + "/Train.txt"
TestFile = Folder + "/Test.txt"

Existing = [f for f in (TrainFile, TestFile) if os.path.exists(f)]
if Existing and not Overwrite:
    print(f"Refusing to overwrite the existing dataset in '{Folder}':")
    for f in Existing:
        print(f"  {f}")
    print("These files are the ones the shipped Info.txt and QUBO files were")
    print("generated from. Regenerating the dataset reshuffles it and can change")
    print("the verified input and the perturbable pixels, which would invalidate")
    print("the shipped QUBO instances and results.")
    print("Re-run with OVERWRITE=1 python DatasetCreation.py if that is intended.")
    raise SystemExit(1)

train_dataloader, test_dataloader = dataloaders(args)

Datasize  = train_dataloader.dataset[0][0].shape
try:
    os.makedirs(Folder)
    print(f"Folder '{Folder}' created successfully!")
except FileExistsError:
    print(f"Folder '{Folder}' already exists!")

torch.save(train_dataloader, TrainFile)
torch.save(test_dataloader, TestFile)

print(len(train_dataloader.dataset))
print(len(test_dataloader.dataset))


file = open(Folder+"/Info.txt", "w")
file.write(f"Batch Size : {batch_size_train} \n" )
file.write(f"Picture Size : {InputSize}x{InputSize} \n" )
file.write(f"Data Size : {Datasize[0]} \n" )
file.write(f"Shuffle : {args.shuffle} \n" )
file.write(f"Remove Contracdicting : {args.remove_contradicting} \n" )
file.write(f"Adpative : {args.use_adaptive}")
file.close()