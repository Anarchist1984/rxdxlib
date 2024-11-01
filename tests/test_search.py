import pytest
from rxdxlib.preprocessing.searcher import MoleculeSearcher

@pytest.fixture
def searcher():
    return MoleculeSearcher(threshold=80, max_records=100, max_attempts=5, sleep_time=1)

def test_query_pubchem_for_similar_compounds(searcher, requests_mock):
    smiles = "CCO"
    mock_key = "mock_key"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/smiles/{smiles}/JSON?Threshold=80&MaxRecords=100"
    requests_mock.get(url, json={"Waiting": {"ListKey": mock_key}})
    
    key = searcher.query_pubchem_for_similar_compounds(smiles)
    assert key == mock_key

def test_check_and_download(searcher, requests_mock):
    key = "mock_key"
    mock_cids = [1, 2, 3]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{key}/cids/JSON"
    requests_mock.get(url, json={"IdentifierList": {"CID": mock_cids}})
    
    cids = searcher.check_and_download(key)
    assert cids == mock_cids

def test_smiles_from_pubchem_cids(searcher, requests_mock):
    cids = [1, 2, 3]
    mock_smiles = ["CCO", "CCC", "CCN"]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, cids))}/property/CanonicalSMILES/JSON"
    requests_mock.get(url, json={"PropertyTable": {"Properties": [{"CanonicalSMILES": smiles} for smiles in mock_smiles]}})
    
    smiles = searcher.smiles_from_pubchem_cids(cids)
    assert smiles == mock_smiles

def test_query_pubchem_for_similar_compounds_invalid_smiles(searcher, requests_mock):
    smiles = "invalid_smiles"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/smiles/{smiles}/JSON?Threshold=80&MaxRecords=100"
    requests_mock.get(url, status_code=400)
    
    with pytest.raises(requests.exceptions.HTTPError):
        searcher.query_pubchem_for_similar_compounds(smiles)

def test_check_and_download_no_matches(searcher, requests_mock):
    key = "mock_key"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{key}/cids/JSON"
    requests_mock.get(url, json={})
    
    with pytest.raises(ValueError):
        searcher.check_and_download(key)