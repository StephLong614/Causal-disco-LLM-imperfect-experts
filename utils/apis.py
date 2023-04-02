import pymed
from pymed import PubMed

def get_abstracts(query, n=1):
    # Create a PubMed object
    pubmed = PubMed(tool="MyTool", email="myemail@domain.com")
    
    # Execute the query
    results = pubmed.query(query, max_results=n)
    
    abstracts = [article.abstract for article in results]
    
    return abstracts
