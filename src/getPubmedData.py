from Bio import Entrez
import json

Entrez.email = "avisave706@gmail.com"  # Always set your email

query = "skin cancer"  # Example query
max_results = 100  # Maximum number of results to fetch

def fetch_pubmed(query, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    papers = []
    for pmid in id_list:
        fetch_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
        fetch_record = Entrez.read(fetch_handle)
        article = fetch_record['PubmedArticle'][0]['MedlineCitation']['Article']
        title = article.get('ArticleTitle', '')
        abstract = article.get('Abstract', {}).get('AbstractText', [''])[0]
        papers.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract
        })
    return papers

# Example usage
papers = fetch_pubmed(query, max_results=max_results)
with open("pubmed_papers.json", "w") as f:
    json.dump(papers, f, indent=2)