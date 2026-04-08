## 1. Principles
Whatever the remapping schema, the labelset used in all plans must be valid. If a plan is using imported labels and others are not, the imported labels cannot be stored as new labels .

### a. Remappings can never be a pure [src],[dest] list, onlyu time lists are allowed if there are multiple datasources and then there is a list of remappings , one per datasource

## 2. Post-YOLO baseline improvements for 2D CT topogram multi-class bbox detection

After a first YOLO baseline, the highest-yield improvements are usually not just replacing it with a newer YOLO version.

### a. Strengthen evaluation first
Use patient-level splits, then add an external test split separated by scanner, vendor, site, and projection view if possible. Review hard negatives explicitly, especially overlap, bowel gas, truncation, anatomy lookalikes, and metallic devices.

### b. Improve the input representation
For topograms, test image representations beyond raw grayscale only:
- soft-tissue window
- bone window
- CLAHE or luminance-adapted image
- body-mask-normalized image

### c. Prioritise small-object and rare-class handling
If some classes are small or infrequent, test:
- higher input resolution
- tiling or cropping around the body region
- rare-class oversampling
- class-balanced sampling
- focal-style losses
- per-class confidence thresholds

### d. Benchmark one strong non-YOLO detector
Do not assume YOLO is the performance ceiling. Run at least one stronger alternative detector as a reference point.

### e. Fuse AP and lateral topograms if both exist
If both scout projections are available for the same study, add study-level multi-view fusion. This is likely one of the most task-specific improvements for topograms.

### f. Use downstream CT as a teacher when paired CT exists
If the topogram is paired with the acquired CT, use CT-derived labels, segmentations, or pseudo-labels projected back into scout space. This can reduce annotation noise in the 2D projection domain.

### g. Add one auxiliary task
Useful auxiliary tasks include:
- study-level class presence classification
- anatomy-region classification
- rough scan-range localisation

### h. Optimise for medical operating points, not only mAP
Track:
- sensitivity at fixed false positives per image or study
- per-class recall
- calibration
- missed clinically important cases

### i. Practical experiment order
1. External or vendor-stratified evaluation plus error taxonomy.
2. Body crop plus multi-window input.
3. Rare-class reweighting or oversampling plus higher resolution.
4. One non-YOLO benchmark detector.
5. AP and lateral fusion, or a CT-teacher pipeline if paired CT is available.

### j. Related literature pointers
- CT localizer detection and model comparison: https://pubmed.ncbi.nlm.nih.gov/40316718/
- Multi-institution multi-vendor CT scout detection: https://pubmed.ncbi.nlm.nih.gov/40930679/
- Luminance adaptation with YOLO in projection-style imaging: https://pubmed.ncbi.nlm.nih.gov/34067462/
- YOLOv8 detector improvement paper: https://pubmed.ncbi.nlm.nih.gov/40893530/
- Segmentation and feature-upgrade detector paper: https://pubmed.ncbi.nlm.nih.gov/40918074/
- Two-view scout reconstruction context: https://pubmed.ncbi.nlm.nih.gov/34908175/
- Automatic positioning from AP localizers: https://pubmed.ncbi.nlm.nih.gov/36703015/
- Cross-modality weak supervision context: https://pubmed.ncbi.nlm.nih.gov/37652014/
- Multi-objective CT detection context: https://pubmed.ncbi.nlm.nih.gov/40232589/
