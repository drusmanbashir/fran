import torch
from monai.data import MetaTensor

  def make_random_fran_batch(
      batch_size=2,
      in_channels=1,
      spatial_size=(160, 160, 64),
      num_classes=3,
  ):
      image = torch.randn(batch_size, in_channels, *spatial_size, dtype=torch.float32)
      lm = torch.randint(
          low=0,
          high=num_classes,
          size=(batch_size, 1, *spatial_size),
          dtype=torch.int64,
      )

      image_meta = {
          "filename_or_obj": [
              f"/tmp/fake_case_{i:04d}_image.nii.gz" for i in range(batch_size)
          ]
      }
      lm_meta = {
          "filename_or_obj": [
              f"/tmp/fake_case_{i:04d}_lm.nii.gz" for i in range(batch_size)
          ]
      }

      batch = {
          "image": MetaTensor(image, meta=image_meta),
          "lm": MetaTensor(lm, meta=lm_meta),
      }
      return batch

if __name__ == '__main__':
    

  image = torch.load("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/images/kits21_00002.pt", weights_only=False)
  lm = torch.load("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms/kits21_00002.pt", weights_only=False)

  batch = {
      "image": MetaTensor(
          image[:1, :1, :160, :160, :64].float(),
          meta={"filename_or_obj": ["/tmp/test_image.nii.gz"]},
      ),
      "lm": MetaTensor(
          lm[:1, :1, :160, :160, :64].long(),
          meta={"filename_or_obj": ["/tmp/test_lm.nii.gz"]},
      ),
  }
