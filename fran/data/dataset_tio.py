from fran.data.dataset import *
class DataLoaderForTIO(DataLoader):

    # def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
    def do_batch(self, b):
        return self.create_batch(self.before_batch(b))

    def create_batch(self, b):
        try:
            return (fa_collate, fa_convert)[self.prebatched](b)
        except Exception as e:
            if not self.prebatched:
                collate_error(e, b)
            raise


# %%
class ImageMaskBBoxDatasetWrapper(SubjectsDataset, ImageMaskBBoxDataset):
    def __init__(self, case_ids, bbox_fn, ensure=["tumour", "cyst"]):
        self.case_ids = case_ids
        ImageMaskBBoxDataset.__init__(
            self, case_ids=case_ids, bbox_fn=bbox_fn, ensure=ensure
        )
        bboxes = load_dict(bbox_fn)
        subjects = []
        self.n_labels = 3

        for bbox in bboxes:
            mask_fn = bbox["filename"]
            img_fn = Path(str(mask_fn).replace("masks", "images"))
            # subjects.append(Subject(image=tio.ScalarImage(img_fn, reader=self.reader), mask = tio.LabelMap(mask_fn,reader=self.reader), bbox = bbox, case_id = bbox['case_id']))
            subjects.append(
                Subject(
                    A=tio.ScalarImage(img_fn, reader=self.reader),
                    B=tio.ScalarImage(mask_fn, reader=self.reader),
                )
            )

        SubjectsDataset.__init__(self, subjects=subjects)

    # def __len__(self):
    #     return ImageMaskBBoxDataset.__len__(self)

    # def __getitem__(self,idx):
    #     tr()
    #     case_id = self.case_ids[idx]
    #     subjects_same_id = [subject for subject in  self._subjects  if subject.case_id == case_id ]
    #
    #     tot_cases= len(subjects_same_id)
    #     if tot_cases>1:
    #         sub_idx = np.random.randint(0,subjects_same_id)
    #     else : sub_idx=0
    #     subject= subjects_same_id[sub_idx]
    #     subject = copy.deepcopy(subject)  # cheap since images not loaded yet
    #     if self.load_getitem:
    #         subject.load()

    def reader(self, path):
        tnsr = torch.load(path)
        tnsr.unsqueeze_(0)
        affine = torch.eye(4)
        return tnsr, affine

def fa_convert_ub(t):
    "A replacement for PyTorch `default_convert` which maintains types and handles `Sequence`s"
    return default_convert(t)  # if isinstance(t, _collate_types)
    # else type(t)([fa_convert(s) for s in t]) if isinstance(t, Sequence)
    # else default_convert(t))




