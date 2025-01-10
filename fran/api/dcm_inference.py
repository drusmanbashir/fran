# %%
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt



import pandas as pd
from label_analysis.helpers import get_labels, to_cc, to_int
from dicom_utils.sitk_to_dcm import nifti_rgb_to_dicom_series 
import os
from label_analysis.merge import LabelMapGeometry
import numpy as np
import pydicom
from pydicom.uid import generate_uid
import nibabel as nib
import nibabel as nib
import pydicom
import os
from dicom_utils.dcm_to_sitk import DCMCaseToSITK
from label_analysis.helpers import to_label
from fran.managers import Project
import ipdb

from fran.inference.cascade import CascadeInferer
from fran.utils.imageviewers import ImageMaskViewer, view_sitk

tr = ipdb.set_trace

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from monai.apps.detection.transforms.array import *
from monai.data.box_utils import *
from monai.inferers.merger import *
from monai.transforms import (AsDiscreted, Compose, Invertd)
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import (Activationsd,
                                              MeanEnsembled)
from monai.transforms.spatial.dictionary import Resized
# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped

from fran.data.dataset import FillBBoxPatchesd
from fran.inference.base import (BaseInferer, get_patch_spacing, 
                                 list_to_chunks, load_params)
from fran.transforms.inferencetransforms import (
    BBoxFromPTd, KeepLargestConnectedComponentWithMetad, RenameDictKeys,
    SaveMultiChanneld, ToCPUd)
from fran.utils.itk_sitk import *

sys.path += ["/home/ub/code"]

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
import sys
from pathlib import Path



def overlay_img_pred(pred_fn,img_fn, out_fn):
    pred = sitk.ReadImage(pred_fn)
    pred = to_label(pred)
    img = sitk.ReadImage(img_fn)

    LO = sitk.LabelMapContourOverlayImageFilter()
    overlaid_nii= LO.Execute(pred,img)
    print("Writing to",out_fn)
    sitk.WriteImage(overlaid_nii,out_fn)




if __name__ == '__main__':
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    input_fldr = Path("/home/ub/Desktop/10_CT_1/4")
    output_fldr = Path("/home/ub/Desktop/10_CT_1/nifti")
    # case_id = "crc_CRC089"
    imgs_tmp = ["/home/ub/Desktop/10_CT_1/nifti/4.nii.gz"]
    dcm_fldr = Path("/home/ub/Desktop/10_CT_1/4/DICOM")
    # preds = En.run(imgs_tmp,chunksize=1)
    overlay_nii_file = "/home/ub/Desktop/10_CT_1/4/overlay/4.nii.gz"
    pred_file = "/s/fran_storage/predictions/litsmc/LITS-933/4.nii.gz"
    output_directory = Path("/home/ub/Desktop/10_CT_1/4/DICOM_OVERLAY")



# %%
#SECTION:-------------------- DICOM to SITK--------------------------------------------------------------------------------------

    D = DCMCaseToSITK(dataset_name="lits", case_folder=input_fldr,output_folder=output_fldr,max_series_per_case=1)
    D.process()
    img_fns = D.output_names
# %%
#SECTION:-------------------- RUN PREDICTIONS --------------------------------------------------------------------------------------
# %%

    run_litsmc= ["LITS-933"]
    run = run_litsmc
    localiser_labels_litsmc = [3]
    run_w = "LITS-1088"
    devices = [1]
    overwrite=True
    safe_mode=True
    save_localiser=True
    save_channels=False
    project = Project(project_title="litsmc")
    if project.project_title=="litsmc":
        k_largest= 1
    else:
        k_largest= None
    En = CascadeInferer(
        run_w,
        run,
        save_channels=save_channels,
        devices=devices ,
        overwrite=overwrite,
        localiser_labels=localiser_labels_litsmc,
        safe_mode=safe_mode,
        save_localiser=save_localiser,
        k_largest=k_largest
    )
# %%
#SECTION:-------------------- OVERLAY PRED CONTOUR OVER IMG--------------------------------------------------------------------------------------
    overlay_img_pred(pred_file,imgs_tmp[0],overlay_nii_file)


# %%
# %%
#SECTION:-------------------- OVERLAY NII TO DICOM--------------------------------------------------------------------------------------
    nifti_rgb_to_dicom_series(
        nifti_path=overlay_nii_file,
        ref_dicom_dir=dcm_fldr,
        output_dir=output_directory,
        series_description="My RGB Series",
    )
# %%
# %%
#SECTION:-------------------- SUMMARY REPORT PDF--------------------------------------------------------------------------------------
    lm = sitk.ReadImage(pred_file)
    overlay_img = sitk.ReadImage(overlay_nii_file)
    L = LabelMapGeometry(lm=lm,ignore_labels=[1])
# Keep only specified columns and rename labels
    L.nbrhoods = L.nbrhoods.rename(columns={'cent': 'centroid'})
    L.nbrhoods = L.nbrhoods[['label', 'volume', 'length', 'centroid','bbox']].copy()
    L.nbrhoods['label'] = L.nbrhoods['label'].replace({3: 'tumour', 2: 'benign'})
    print(L.nbrhoods)

# %%
    lm = to_int ( lm)
    LS = sitk.LabelShapeStatisticsImageFilter()
    LS.ComputeFeretDiameterOn()
    LS.Execute(lm)
    cent= LS.GetCentroid(1)
    volume =LS.GetPhysicalSize(1)*1e-3
    length = LS.GetFeretDiameter(1)

    bbox = LS.GetBoundingBox(1)

# Add liver measurements as new row to nbrhoods dataframe
# %%
    liver_row = pd.DataFrame({
        'label': ['liver'],
        'volume': [volume],
        'length': [length],
        'centroid': [cent],
        'bbox':[bbox]
    })
# %%
    L.nbrhoods = pd.concat([L.nbrhoods, liver_row], ignore_index=True)

# %%
# Create the PDF report
    pdf_path = os.path.join(output_directory.parent, 'report.pdf')


    L.nbrhoods = L.nbrhoods.drop(columns=['centroid'])
    L.nbrhoods = L.nbrhoods.rename(columns={'length': 'diameter'})
    L.nbrhoods = pd.concat([L.nbrhoods, liver_row], ignore_index=True)

    # Create the PDF report
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Save slice as temporary image first (we need it for the layout)
# %%
#SECTION:-------------------- CREATE 2d figure--------------------------------------------------------------------------------------

    # Get appropriate slice from overlay image
    overlay_array = sitk.GetArrayFromImage(overlay_img)
    
    # Determine slice location based on tumour presence
    tumour_data = L.nbrhoods[L.nbrhoods['label'] == 'tumour']
    if not tumour_data.empty:
        # Use the first tumour's centroid z-coordinate
        tumour_bbox = tumour_data.iloc[0]['bbox']

        slice_idx = tumour_bbox[2] + int(tumour_bbox[-1]/2)
    else:
        # Use middle liver slice if no tumour

        liver_data= L.nbrhoods[L.nbrhoods['label'] == 'liver']
        liver_bbox = liver_data.iloc[0]['bbox']
        slice_idx = liver_bbox[2] + int(liver_bbox[-1]/2)
    
    mid_slice = overlay_array[slice_idx]
    # Create figure without displaying it
    fig = plt.figure(figsize=(10, 10))
    plt.ioff()  # Turn off interactive mode
    ax = fig.add_subplot(111)
    ax.imshow(mid_slice)
    ax.axis('off')
    temp_img_path = os.path.join(output_directory, 'temp_slice.png')
    fig.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# %%
    # Add title

    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph('Liver Analysis Report', styles['Heading1']))
    elements.append(Spacer(1, 20))
    
    # Create a table for the top section (Liver Stats and Image side by side)
    liver_data = L.nbrhoods[L.nbrhoods['label'] == 'liver'].iloc[0]
    
    # Create left column content as a single cell with multiple paragraphs
    left_content = []
    left_content.append(Paragraph('Liver Statistics', styles['Heading2']))
    left_content.append(Spacer(1, 10))
    left_content.append(Paragraph(f"Volume: {liver_data['volume']:.1f} cm³", styles['Normal']))
    left_content.append(Spacer(1, 5))
    left_content.append(Paragraph(f"Diameter: {liver_data['diameter']:.1f} mm", styles['Normal']))
    
    # Create the table with a single row
    liver_stats = [[
        left_content,  # Left cell with all text content
        Image(temp_img_path, width=200, height=200)  # Right cell with image
    ]]
    
    top_table = Table(liver_stats, colWidths=[300, 200])
    top_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ('LEFTPADDING', (0, 0), (0, 0), 0),  # Remove left padding for text
    ('TOPPADDING', (0, 0), (-1, -1), 0),  # Remove top padding
    ]))
    elements.append(top_table)
    elements.append(Spacer(1, 20))

    # Tumour Statistics (if present)
    tumour_data = L.nbrhoods[L.nbrhoods['label'] == 'tumour']
    if not tumour_data.empty:
        elements.append(Paragraph('Tumour Statistics', styles['Heading2']))
        num_lesions = len(tumour_data)
        elements.append(Paragraph(f"Total number of lesions: {num_lesions}", styles['Normal']))
        elements.append(Spacer(1, 10))
        
        # Add index column to tumour table
        tumour_table_data = [['#', 'Volume (cm³)', 'Diameter (mm)']]
        for idx, row in tumour_data.iterrows():
            tumour_table_data.append([
                str(idx + 1),
                f"{row['volume']:.1f}",
                f"{row['diameter']:.1f}"
            ])
        
        tumour_table = Table(tumour_table_data)
        tumour_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(tumour_table)
        elements.append(Spacer(1, 20))
# %%
# %%
    
    # Save slice as temporary image
    
    # Add image to PDF
    # Generate PDF
    doc.build(elements)
    
    # Clean up temporary image
    os.remove(temp_img_path)
    print(f"PDF report generated at: {pdf_path}")

# %%




