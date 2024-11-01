import time
from pathlib import Path
from urllib.parse import quote
import warnings

from IPython.display import Markdown, Image
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


class MoleculeSearcher:
    def __init__(self, threshold=80, max_records=100, max_attempts=50, sleep_time=5):
        """
        Initialize the molecule searcher.
        
        Parameters
        ----------
        threshold : int
            Similarity threshold (0-100)
        max_records : int
            Maximum number of records to retrieve
        max_attempts : int
            Maximum number of attempts to check job status
        sleep_time : int
            Sleep time between job status checks in seconds
        """
        self.threshold = threshold
        self.max_records = max_records
        self.max_attempts = max_attempts
        self.sleep_time = sleep_time
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def query_pubchem_for_similar_compounds(self, smiles, threshold=None, n_records=None):
        """
        Query PubChem for similar compounds and return the job key.

        Parameters
        ----------
        smiles : str
            The canonical SMILES string for the given compound.
        threshold : int
            The threshold of similarity, default 75%. In PubChem, the default threshold is 90%.
        n_records : int
            The maximum number of feedback records.

        Returns
        -------
        str
            The job key from the PubChem web service.
        """
        threshold = threshold or self.threshold
        n_records = n_records or self.max_records
        
        escaped_smiles = quote(str(smiles))  # Ensure smiles is a string and properly encoded
        url = f"{self.base_url}/compound/similarity/smiles/{escaped_smiles}/JSON?Threshold={threshold}&MaxRecords={n_records}"
        r = requests.get(url)
        r.raise_for_status()
        key = r.json()["Waiting"]["ListKey"]
        return key
    
    def check_and_download(self, key, attempts=None):
        """
        Check job status and download PubChem CIDs when the job finished

        Parameters
        ----------
        key : str
            The job key of the PubChem service.
        attempts : int
            The time waiting for the feedback from the PubChem service.

        Returns
        -------
        list
            The PubChem CIDs of similar compounds.
        """
        attempts = attempts or self.max_attempts
        url = f"{self.base_url}/compound/listkey/{key}/cids/JSON"
        print(f"Querying for job {key} at URL {url}...", end="")
        while attempts:
            r = requests.get(url)
            r.raise_for_status()
            response = r.json()
            if "IdentifierList" in response:
                cids = response["IdentifierList"]["CID"]
                return cids
            attempts -= 1
            print(".", end="")
            time.sleep(self.sleep_time)
        raise ValueError(f"Could not find matches for job key: {key}")
    
    def smiles_from_pubchem_cids(self, cids, batch_size=100):
        """
        Get the canonical SMILES string from the PubChem CIDs.
        Handles large numbers of CIDs by processing them in batches.

        Parameters
        ----------
        cids : list
            A list of PubChem CIDs.
        batch_size : int
            Number of CIDs to process in each request.

        Returns
        -------
        list
            The canonical SMILES strings of the PubChem CIDs.
        """
        all_smiles = []
        
        # Process CIDs in batches
        for i in range(0, len(cids), batch_size):
            batch = cids[i:i + batch_size]
            url = f"{self.base_url}/compound/cid/{','.join(map(str, batch))}/property/CanonicalSMILES/JSON"
            r = requests.get(url)
            r.raise_for_status()
            all_smiles.extend([item["CanonicalSMILES"] for item in r.json()["PropertyTable"]["Properties"]])
            
            # Optional: add a small delay between requests to be nice to the server
            time.sleep(0.2)
            
        return all_smiles

query = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
searcher = MoleculeSearcher(threshold=80, max_records=20000)
job_key = searcher.query_pubchem_for_similar_compounds(query)
similar_cids = searcher.check_and_download(job_key)
similar_smiles = searcher.smiles_from_pubchem_cids(similar_cids)


query_results_df = pd.DataFrame({"smiles": similar_smiles, "CIDs": similar_cids})
PandasTools.AddMoleculeColumnToFrame(query_results_df, smilesCol="smiles")
query_results_df