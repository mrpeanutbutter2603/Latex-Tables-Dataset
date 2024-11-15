import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import tarfile
import os

class ArxivClient:
    BASE_URL = 'http://export.arxiv.org/api/query'
    SOURCE_BASE_URL = 'https://arxiv.org/src/'
    
    def __init__(self, delay_seconds=3):
        self.delay_seconds = delay_seconds
    
    def get_by_id(self, paper_id):
        """
        Get paper by arXiv ID
        
        Parameters:
        paper_id (str): arXiv paper ID (e.g. '2401.00001' or 'quant-ph/0201082')
        
        Returns:
        dict: Paper information
        """
        params = {
            'id_list': paper_id,
            'max_results': 1
        }
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        time.sleep(self.delay_seconds)
        
        with urllib.request.urlopen(url) as response:
            tree = ET.parse(response)
            root = tree.getroot()
            
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is None:
                return None
                
            paper = {
                'title': entry.find('atom:title', ns).text.strip(),
                'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                'summary': entry.find('atom:summary', ns).text.strip(),
                'published': entry.find('atom:published', ns).text,
                'arxiv_id': entry.find('atom:id', ns).text.split('/')[-1],
                'pdf_url': entry.find('atom:link[@title="pdf"]', ns).get('href'),
                'categories': [cat.get('term') for cat in entry.findall('arxiv:primary_category', ns)]
            }
            return paper
    
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
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Construct source URL
            source_url = f"{self.SOURCE_BASE_URL}{paper_id}"
            
            # Temporary file for download
            temp_file = os.path.join(output_dir, f"{paper_id}.tar.gz")
            
            # Download the file
            print(f"Downloading source from: {source_url}")
            time.sleep(self.delay_seconds)
            urllib.request.urlretrieve(source_url, temp_file)
            
            # Create extraction directory
            extract_dir = os.path.join(output_dir, paper_id)
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract the tar.gz file
            print(f"Extracting files to: {extract_dir}")
            with tarfile.open(temp_file, "r:gz") as tar:
                # Check for potential path traversal
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
            
            # Clean up the temporary file
            os.remove(temp_file)
            
            # List extracted files
            files = os.listdir(extract_dir)
            tex_files = [f for f in files if f.endswith('.tex')]
            
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

if __name__ == "__main__":
    main()