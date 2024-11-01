import pytest
import pandas as pd
from rxdxlib.preprocessing.preprocessor import mainpreprocessor, inhibitorprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'mw': [300, 500, None, 600],
        'polararea': [20, 30, 40, 50],
        'complexity': [100, 200, 300, 400],
        'xlogp': [2.5, 3.5, 1.5, 2.0],
        'heavycnt': [20, 25, 30, 35],
        'hbonddonor': [2, 3, 1, 4],
        'hbondacc': [3, 2, 4, 1],
        'rotbonds': [4, 5, 3, 2],
        'inchi': ['inchi1', 'inchi2', 'inchi3', 'inchi4'],
        'exactmass': [300.1, 500.1, 600.1, 700.1],
        'monoisotopicmass': [300.1, 500.1, 600.1, 700.1],
        'charge': [0, 0, 0, 0],
        'covalentunitcnt': [1, 1, 1, 1],
        'isotopeatomcnt': [0, 0, 0, 0],
        'totalatomstereocnt': [0, 0, 0, 0],
        'definedatomstereocnt': [0, 0, 0, 0],
        'undefinedatomstereocnt': [0, 0, 0, 0],
        'totalbondstereocnt': [0, 0, 0, 0],
        'definedbondstereocnt': [0, 0, 0, 0],
        'undefinedbondstereocnt': [0, 0, 0, 0],
        'cid': [1, 2, 2, 3]
    })

def test_dropna_transformer(sample_data):
    transformer = mainpreprocessor.DropNA()
    result = transformer.fit_transform(sample_data)
    assert len(result) == 3  # One row should be dropped due to NaN

def test_remove_duplicates_with_same_cid(sample_data):
    transformer = mainpreprocessor.RemoveDuplicatesWithSameCID(cid_column='cid')
    result = transformer.fit_transform(sample_data)
    assert len(result) == 3  # One duplicate (cid=2) should be removed

def test_drop_duplicates(sample_data):
    transformer = mainpreprocessor.DropDuplicates()
    result = transformer.fit_transform(sample_data)
    assert len(result) == 4  # No duplicates in original data, so length stays the same

def test_keep_necessary_columns(sample_data):
    transformer = mainpreprocessor.KeepNecessaryColumns()
    result = transformer.fit_transform(sample_data)
    expected_columns = transformer.columns
    assert list(result.columns) == expected_columns  # Verify only necessary columns are retained
    assert len(result) == len(sample_data)  # Verify row count stays the same

def test_scale_data(sample_data):
    transformer = mainpreprocessor.ScaleData()
    transformer.fit(sample_data)
    result = transformer.transform(sample_data)
    
    # Verify that non-'inchi' columns are scaled and 'inchi' is intact
    assert 'inchi' in result.columns
    assert not result[['mw', 'polararea', 'complexity']].equals(sample_data[['mw', 'polararea', 'complexity']])
    assert (result['inchi'] == sample_data['inchi']).all()  # Verify 'inchi' column remains unchanged

def test_pipeline(sample_data):
    preprocessor = mainpreprocessor()
    result = preprocessor.process(sample_data)
    
    # Ensure that the pipeline returns the expected number of rows and columns
    assert len(result) == 3  # One row should be dropped (due to NaN), one duplicate removed
    assert len(result.columns) == 20  # Only necessary columns should remain

def test_inhibitor_processor(sample_data):
    inhibitor_value = 1
    processor = inhibitorprocessor(inhibitor_value)
    result = processor.process(sample_data)
    
    # Check if 'inhibitor' column is added with correct values
    assert 'inhibitor' in result.columns
    assert (result['inhibitor'] == inhibitor_value).all()
    assert len(result) == 3  # Same logic as the main pipeline (drop NaN and duplicates)

def test_noninhibitor_processor(sample_data):
    inhibitor_value = 0  # Non-inhibitor value
    processor = inhibitorprocessor(inhibitor_value)
    result = processor.process(sample_data)
    
    # Check if 'inhibitor' column is added with correct values (0 for non-inhibitor)
    assert 'inhibitor' in result.columns
    assert (result['inhibitor'] == inhibitor_value).all()  # All values in 'inhibitor' column should be 0
    assert len(result) == 3  # One row should be dropped (due to NaN), one duplicate removed

def test_invalid_input_type():
    preprocessor = mainpreprocessor()
    
    # Test passing non-DataFrame input
    with pytest.raises(ValueError):
        preprocessor.process([1, 2, 3])

def test_invalid_cid_column(sample_data):
    # Test RemoveDuplicatesWithSameCID with invalid cid column
    transformer = mainpreprocessor.RemoveDuplicatesWithSameCID(cid_column='invalid_cid')
    with pytest.raises(ValueError):
        transformer.transform(sample_data)

def test_missing_columns(sample_data):
    # Test KeepNecessaryColumns with missing columns
    sample_data_missing_cols = sample_data.drop(columns=['mw', 'polararea'])
    transformer = mainpreprocessor.KeepNecessaryColumns()
    with pytest.raises(ValueError):
        transformer.transform(sample_data_missing_cols)