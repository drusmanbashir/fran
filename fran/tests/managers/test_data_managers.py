import pytest
import torch
from pathlib import Path
from fran.managers.project import Project
from fran.configs.parser import ConfigMaker
from fran.utils.common import *
from utilz.helpers import load_dict
from utilz.imageviewers import ImageMaskViewer
from fran.managers.data.training import (
    DataManagerWhole,
    DataManagerBaseline,
    DataManagerSource,
    DataManagerLBD,
    DataManagerPatch
)

@pytest.fixture
def litsmc_project():
    return Project(project_title="litsmc")

@pytest.fixture
def totalseg_project():
    return Project(project_title="totalseg")

@pytest.fixture
def litsmc_config(litsmc_project):
    return ConfigMaker(litsmc_project, ).config

@pytest.fixture
def totalseg_config(totalseg_project):
    return ConfigMaker(totalseg_project, ).config

class TestDataManagerWhole:
    def test_initialization(self, totalseg_project, totalseg_config):
        dm = DataManagerWhole(
            project=totalseg_project,
            batch_size=4,
            config=totalseg_config,
        )
        assert dm is not None
        assert dm.batch_size == 4

    def test_data_preparation(self, totalseg_project, totalseg_config):
        dm = DataManagerWhole(
            project=totalseg_project,
            batch_size=4,
            config=totalseg_config,
        )
        dm.prepare_data()
        dm.setup()
        
        # Test dataloader creation
        dl = dm.train_dataloader()
        assert dl is not None
        
        # Test dataset access
        batch = dm.train_ds[0]
        assert 'image' in batch
        assert 'lm' in batch

class TestDataManagerBaseline:
    def test_initialization(self, totalseg_project, totalseg_config):
        dm = DataManagerBaseline(
            project=totalseg_project,
            config=totalseg_config,
            batch_size=2
        )
        assert dm is not None
        assert dm.batch_size == 2

    def test_data_preparation(self, totalseg_project, totalseg_config):
        dm = DataManagerBaseline(
            project=totalseg_project,
            config=totalseg_config,
            batch_size=2
        )
        dm.prepare_data()
        dm.setup()
        
        # Verify batch size limitation
        assert len(dm.data_train) == dm.batch_size
        
        # Test dataset sample
        batch = dm.train_ds[0]
        assert 'image' in batch
        assert 'lm' in batch

class TestDataManagerSource:
    def test_initialization(self, totalseg_project, totalseg_config):
        dm = DataManagerSource(
            project=totalseg_project,
            config=totalseg_config,
            batch_size=2
        )
        assert dm is not None
        assert hasattr(dm, 'effective_batch_size')

    def test_data_preparation(self, totalseg_project, totalseg_config):
        dm = DataManagerSource(
            project=totalseg_project,
            config=totalseg_config,
            batch_size=2
        )
        dm.prepare_data()
        dm.setup()
        
        # Test dataloader
        dl = dm.train_dataloader()
        assert dl is not None

class TestDataManagerLBD:
    def test_initialization(self, totalseg_project, totalseg_config):
        dm = DataManagerLBD(
            project=totalseg_project,
            config=totalseg_config,
            batch_size=2
        )
        assert dm is not None
        assert hasattr(dm, 'effective_batch_size')

    def test_data_preparation(self, totalseg_project, totalseg_config):
        dm = DataManagerLBD(
            project=totalseg_project,
            config=totalseg_config,
            batch_size=2
        )
        dm.prepare_data()
        dm.setup()
        
        # Test data folder derivation
        assert dm.data_folder is not None
        assert dm.data_folder.exists()

class TestDataManagerPatch:
    def test_initialization(self, litsmc_project, litsmc_config):
        dm = DataManagerPatch(
            project=litsmc_project,
            config=litsmc_config,
            batch_size=2
        )
        assert dm is not None

    def test_data_preparation(self, litsmc_project, litsmc_config):
        dm = DataManagerPatch(
            project=litsmc_project,
            config=litsmc_config,
            batch_size=2
        )
        dm.prepare_data()
        dm.setup()
        
        # Test bboxes loading
        assert hasattr(dm, 'bboxes')
        assert dm.bboxes is not None
        
        # Test dataloader
        dl = dm.train_dataloader()
        assert dl is not None

if __name__ == "__main__":
    pytest.main([__file__])


