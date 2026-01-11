## 1. Patch size vs src_dims

Patch size is used by transform 
src_dims only in the train dataloader (not valid dataloader) By transform Rtr which gets a random patch, next in line Re (ResizeWithPadOrCrop) zooms in into this presample.

### 1.1 Validation
This is GridPatch dataset. Iterates whole dataset, whatever the mode of the plan_train, this mode is fixed and set as a plan_valid setting in dataset_params. However, patch_size is copied over from plan_train during config parser
