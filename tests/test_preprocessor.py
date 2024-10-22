import pandas as pd
from sklearn.preprocessing import StandardScaler
from rxdxlib.preprocessing.preprocessor import mainpreprocessor

# Test DropNA Transformer
def test_dropna_with_na():
    data_with_na = pd.DataFrame({
        'mw': [300.0, None],
        'polararea': [50.0, 55.0]
    })
    transformer = mainpreprocessor.DropNA()
    result = transformer.transform(data_with_na)
    assert result.shape == (1, 2)  # One row should be dropped due to NaN

# Test RemoveDuplicatesWithSameCID Transformer
def test_remove_duplicates_with_same_cid():
    data = pd.DataFrame({
        'cid': [1, 2, 2, 3],
        'value': [10, 20, 20, 30]
    })
    processor = mainpreprocessor()  # Initialize the main preprocessor
    result = processor.pipeline.named_steps['remove_duplicates_with_same_cid'].transform(data)
    
    expected = pd.DataFrame({
        'cid': [1, 2, 3],
        'value': [10, 20, 30]
    }).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

# Test DropDuplicates Transformer
def test_drop_duplicates():
    data = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6]
    })
    processor = mainpreprocessor()
    result = processor.pipeline.named_steps['drop_duplicates'].transform(data)
    
    expected = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

# Test KeepNecessaryColumns Transformer
def test_keep_necessary_columns():
    data = pd.DataFrame({
        'mw': [1, 2],
        'polararea': [3, 4],
        'complexity': [5, 6],
        'xlogp': [7, 8],
        'heavycnt': [9, 10],
        'hbonddonor': [11, 12],
        'hbondacc': [13, 14],
        'rotbonds': [15, 16],
        'inchi': ['a', 'b'],
        'exactmass': [17, 18],
        'monoisotopicmass': [19, 20],
        'charge': [21, 22],
        'covalentunitcnt': [23, 24],
        'isotopeatomcnt': [25, 26],
        'totalatomstereocnt': [27, 28],
        'definedatomstereocnt': [29, 30],
        'undefinedatomstereocnt': [31, 32],
        'totalbondstereocnt': [33, 34],
        'definedbondstereocnt': [35, 36],
        'undefinedbondstereocnt': [37, 38],
        'extra_column': [39, 40]  # Extra column not needed
    })
    
    processor = mainpreprocessor()
    result = processor.pipeline.named_steps['keep_necessary_columns'].transform(data)
    
    expected = data.drop(columns=['extra_column']).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

# Test ScaleData Transformer
def test_scale_data():
    data = pd.DataFrame({
        'mw': [1, 2],
        'polararea': [3, 4],
        'complexity': [5, 6],
        'xlogp': [7, 8],
        'heavycnt': [9, 10],
        'hbonddonor': [11, 12],
        'hbondacc': [13, 14],
        'rotbonds': [15, 16],
        'inchi': ['a', 'b'],  # 'inchi' should not be scaled
        'exactmass': [17, 18],
        'monoisotopicmass': [19, 20],
        'charge': [21, 22],
        'covalentunitcnt': [23, 24],
        'isotopeatomcnt': [25, 26],
        'totalatomstereocnt': [27, 28],
        'definedatomstereocnt': [29, 30],
        'undefinedatomstereocnt': [31, 32],
        'totalbondstereocnt': [33, 34],
        'definedbondstereocnt': [35, 36],
        'undefinedbondstereocnt': [37, 38]
    })
    
    processor = mainpreprocessor()
    result = processor.pipeline.named_steps['scale_data'].fit_transform(data)
    
    # Manually scale the data (excluding 'inchi')
    expected = data.drop(columns=['inchi']).reset_index(drop=True)
    scaler = StandardScaler()
    scaled_expected = pd.DataFrame(scaler.fit_transform(expected), columns=expected.columns).reset_index(drop=True)
    scaled_expected['inchi'] = data['inchi']  # Add back the 'inchi' column, which should not be scaled
    
    pd.testing.assert_frame_equal(result.reset_index(drop=True), scaled_expected)
