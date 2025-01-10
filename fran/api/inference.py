# %%
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import shutil
import SimpleITK as sitk
from typing import List
import torch

from inference.cascade import CascadeInferer
from fran.utils.fileio import maybe_makedirs

app = FastAPI()

MODEL_CONFIGS = {
    "litsmc": {
        "run_w": "LITS-1088",
        "run": ["LITS-933"],
        "localiser_labels": [3],
        "k_largest": 1
    }
    # Add more model configurations as needed
}

def setup_inferer(model_name: str):
    """Setup the CascadeInferer based on model configuration"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown model name: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    
    try:
        inferer = CascadeInferer(
            run_w=config["run_w"],
            runs_p=config["run"],
            localiser_labels=config["localiser_labels"],
            devices=[0],  # Assuming GPU 0
            overwrite=True,
            safe_mode=False,
            save_channels=False,
            k_largest=config.get("k_largest")
        )
        return inferer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup model: {str(e)}")

@app.post("/predict/{model_name}")
async def predict(model_name: str, files: List[UploadFile] = File(...)):
    """
    Endpoint for running inference on uploaded images
    """
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_files = []
        
        # Save uploaded files
        for file in files:
            temp_path = temp_dir / file.filename
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_files.append(temp_path)
        
        try:
            # Setup and run inference
            inferer = setup_inferer(model_name)
            predictions = inferer.run(input_files, chunksize=1)
            
            # Process results
            output_files = []
            for pred in predictions:
                # Save prediction to temp file
                output_path = temp_dir / f"pred_{Path(pred['image'].meta['filename_or_obj']).name}"
                sitk.WriteImage(pred['pred'], str(output_path))
                output_files.append(output_path)
            
            # Return first prediction file (modify as needed for multiple files)
            return FileResponse(
                path=str(output_files[0]),
                filename=output_files[0].name,
                media_type="application/octet-stream"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up CUDA memory
            torch.cuda.empty_cache()

@app.get("/models")
async def list_models():
    """List available model configurations"""
    return {"available_models": list(MODEL_CONFIGS.keys())}

# %%
