import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import tarfile
import os
import http.client
import argparse
from datetime import datetime, timezone

class ArxivClient:
    BASE_URL = 'http://export.arxiv.org/api/query'
    SOURCE_BASE_URL = 'https://arxiv.org/src/'
    
    def __init__(self, delay_seconds=3):
        self.delay_seconds = delay_seconds
    
    def search_cs_papers_with_source(self, max_results=100, start=0, batch_size=100, 
                                   date_start=None, date_end=None):
        """
        Search for computer science papers with source files within a date range
        
        Parameters:
        max_results (int): Maximum number of results
        start (int): Start index for pagination
        batch_size (int): Papers per API call
        date_start (str): Start date in format YYYYMMDDHHMMSS
        date_end (str): End date in format YYYYMMDDHHMMSS
        
        Returns:
        list: Paper information dictionaries with available source files
        """
        papers_with_source = []
        current_start = start
        
        # Construct date range query
        date_query = ''
        if date_start:
            date_query += f' AND submittedDate:[{date_start} TO '
            date_query += date_end if date_end else 'now'
            date_query += ']'
        
        while len(papers_with_source) < max_results:
            params = {
                'search_query': f'cat:cs.*{date_query}',
                'max_results': min(batch_size, max_results - len(papers_with_source)),
                'start': current_start,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            time.sleep(self.delay_seconds)
            
            try:
                with urllib.request.urlopen(url) as response:
                    tree = ET.parse(response)
                    root = tree.getroot()
                    
                    ns = {'atom': 'http://www.w3.org/2005/Atom',
                          'arxiv': 'http://arxiv.org/schemas/atom'}
                    
                    entries = root.findall('atom:entry', ns)
                    if not entries:
                        break
                    
                    for entry in entries:
                        paper = {
                            'title': entry.find('atom:title', ns).text.strip(),
                            'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                            'summary': entry.find('atom:summary', ns).text.strip(),
                            'published': entry.find('atom:published', ns).text,
                            'arxiv_id': entry.find('atom:id', ns).text.split('/')[-1],
                            'pdf_url': entry.find('atom:link[@title="pdf"]', ns).get('href'),
                            'categories': [cat.get('term') for cat in entry.findall('arxiv:primary_category', ns)]
                        }
                        
                        if self.has_source_files(paper['arxiv_id']):
                            papers_with_source.append(paper)
                            print(f"Found paper with source: {paper['arxiv_id']} - {paper['title']}")
                            
                            if len(papers_with_source) >= max_results:
                                break
                        
                        time.sleep(self.delay_seconds)
                    
                    current_start += batch_size
            
            except Exception as e:
                print(f"Error fetching papers: {str(e)}")
                break
        
        return papers_with_source
    
    def has_source_files(self, paper_id):
        """
        Check if source files are available for a paper by making a HEAD request
        
        Parameters:
        paper_id (str): arXiv paper ID
        
        Returns:
        bool: True if source files are available, False otherwise
        """
        source_url = f"{self.SOURCE_BASE_URL}{paper_id}"
        try:
            req = urllib.request.Request(source_url, method='HEAD')
            urllib.request.urlopen(req)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            return False
    
    def batch_download_sources(self, papers, output_dir, max_papers=None):
        """
        Download source files for multiple papers
        
        Parameters:
        papers (list): List of paper dictionaries
        output_dir (str): Base directory for downloads
        max_papers (int): Maximum number of papers to download (None for all)
        
        Returns:
        dict: Dictionary mapping paper IDs to download results
        """
        results = {}
        
        if max_papers is not None:
            papers = papers[:max_papers]
        
        for paper in papers:
            paper_id = paper['arxiv_id']
            print(f"Downloading sources for {paper_id}: {paper['title']}")
            
            success, result = self.download_and_extract_source(paper_id, output_dir)
            results[paper_id] = {
                'success': success,
                'result': result,
                'title': paper['title']
            }
            
            time.sleep(self.delay_seconds)
        
        return results
    
    def download_and_extract_source(self, paper_id, output_dir):
        """
        Download and extract source files from arxiv.org/src
        
        Parameters:
        paper_id (str): arXiv paper ID
        output_dir (str): Directory to extract files to
        
        Returns:
        tuple: (bool, str) - (Success status, Path to extracted files or error message)
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            source_url = f"{self.SOURCE_BASE_URL}{paper_id}"
            temp_file = os.path.join(output_dir, f"{paper_id}.tar.gz")
            
            print(f"Downloading source from: {source_url}")
            time.sleep(self.delay_seconds)
            urllib.request.urlretrieve(source_url, temp_file)
            
            extract_dir = os.path.join(output_dir, paper_id)
            os.makedirs(extract_dir, exist_ok=True)
            
            print(f"Extracting files to: {extract_dir}")
            with tarfile.open(temp_file, "r:gz") as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory
                
                def safe_extract(tar, path):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted path traversal in tar file")
                    tar.extractall(path)
                
                safe_extract(tar, extract_dir)
            
            os.remove(temp_file)
            
            files = os.listdir(extract_dir)
            tex_files = [f for f in files if f.endswith('.tex')]

            # Check for main.tex
            if 'main.tex' not in tex_files:
                print(f"main.tex not found in {extract_dir}. Deleting directory.")
                for root, dirs, files in os.walk(extract_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(extract_dir)
                return False, f"main.tex not found. Directory deleted."

            return True, {
                'extract_dir': extract_dir,
                'files': files,
                'tex_files': tex_files
            }
            
        except Exception as e:
            return False, str(e)

# Example usage
def main():
    client = ArxivClient()
    paper_id = "2411.08982"  # Example paper ID
    
    # Get paper information
    paper = client.get_by_id(paper_id)
    if paper:
        print(f"Downloading source files for: {paper['title']}")
        
        # Download and extract source files
        success, result = client.download_and_extract_source(paper_id, "./arxiv_sources")
        
        if success:
            print("\nExtracted files:")
            print(f"Directory: {result['extract_dir']}")
            print("\nTeX files found:")
            for tex_file in result['tex_files']:
                print(f"- {tex_file}")
            print("\nAll files:")
            for file in result['files']:
                print(f"- {file}")
        else:
            print(f"Error: {result}")

def main_download_cs_100(date_start, date_end, max_results):
    client = ArxivClient()
    print(f"Searching for CS papers with source files from {date_start} to {date_end}...")
    papers = client.search_cs_papers_with_source(
        max_results=max_results,
        date_start=date_start,
        date_end=date_end
    )
    
    if papers:
        print(f"\nFound {len(papers)} papers with source files")
        results = client.batch_download_sources(papers, "./arxiv_sources")
        
        print("\nDownload Results:")
        for paper_id, result in results.items():
            status = "Success" if result['success'] else "Failed"
            print(f"{paper_id} - {status}: {result['title']}")
            if result['success']:
                print(f"  TeX files: {', '.join(result['result']['tex_files'])}")
            else:
                print(f"  Error: {result['result']}")
            print()
    else:
        print("No papers with source files found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for CS papers with source files.")
    parser.add_argument("--date_start", type=str, required=True, help="Start date in the format YYYYMMDDHHMMSS")
    parser.add_argument("--date_end", type=str, required=True, help="End date in the format YYYYMMDDHHMMSS")
    parser.add_argument("--max_results", type=int, default=100, required=True, help="Maximum number of results to return")

    args = parser.parse_args()
    main_download_cs_100(args.date_start, args.date_end, args.max_results)