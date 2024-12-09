import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import tarfile
import os
import argparse
from multiprocessing import Pool, Manager, Lock, Value
from functools import partial

class ArxivClient:
    BASE_URL = 'http://export.arxiv.org/api/query'
    SOURCE_BASE_URL = 'https://arxiv.org/src/'

    def __init__(self, delay_seconds=2):
        self.delay_seconds = delay_seconds

    def search_cs_papers_with_source(self, max_results=100, start=0, batch_size=100, date_start=None, date_end=None):
        """Search for computer science papers with source files within a date range."""
        papers_with_source = []
        current_start = start
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

                    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
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
        """Check if source files are available for a paper."""
        source_url = f"{self.SOURCE_BASE_URL}{paper_id}"
        try:
            req = urllib.request.Request(source_url, method='HEAD')
            urllib.request.urlopen(req)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            return False

    def download_and_extract_source(self, paper_id, output_dir, delay_seconds=None):
        """Download and extract source files."""
        try:
            if delay_seconds:
                time.sleep(delay_seconds)

            os.makedirs(output_dir, exist_ok=True)
            source_url = f"{self.SOURCE_BASE_URL}{paper_id}"
            temp_file = os.path.join(output_dir, f"{paper_id}.tar.gz")

            print(f"Downloading source from: {source_url}")
            urllib.request.urlretrieve(source_url, temp_file)

            extract_dir = os.path.join(output_dir, paper_id)
            os.makedirs(extract_dir, exist_ok=True)

            print(f"Extracting files to: {extract_dir}")
            with tarfile.open(temp_file, "r:gz") as tar:
                tar.extractall(extract_dir)

            os.remove(temp_file)

            tex_files = [f for f in os.listdir(extract_dir) if f.endswith('.tex')]
            if not tex_files:
                print(f"No .tex files found in {extract_dir}. Deleting directory.")
                for root, dirs, files in os.walk(extract_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(extract_dir)
                return False, "No .tex files found. Directory deleted."

            return True, {
                'extract_dir': extract_dir,
                'files': os.listdir(extract_dir),
                'tex_files': tex_files
            }
        except Exception as e:
            return False, str(e)


def download_paper_worker(paper, output_dir, shared_stats, client, delay_seconds):
    """Worker function to download and process a single paper."""
    try:
        paper_id = paper['arxiv_id']

        with shared_stats['lock']:
            shared_stats['current'].value += 1
            current = shared_stats['current'].value
            total = shared_stats['total'].value
            print(f"Processing paper {current}/{total}: {paper_id}")

        success, result = client.download_and_extract_source(
            paper_id, output_dir, delay_seconds=delay_seconds
        )

        with shared_stats['lock']:
            if success:
                shared_stats['successful'].value += 1
            else:
                shared_stats['failed'].value += 1

            progress = (shared_stats['current'].value / shared_stats['total'].value) * 100
            print(f"Progress: {progress:.2f}% ({shared_stats['successful'].value} successful, "
                  f"{shared_stats['failed'].value} failed)")

        return {
            'paper_id': paper_id,
            'success': success,
            'result': result,
            'title': paper['title']
        }
    except Exception as e:
        print(f"Error processing paper {paper['arxiv_id']}: {str(e)}")
        with shared_stats['lock']:
            shared_stats['failed'].value += 1
        return {
            'paper_id': paper['arxiv_id'],
            'success': False,
            'result': str(e),
            'title': paper['title']
        }


def parallel_download_sources(papers, output_dir, num_threads=None, max_papers=None):
    """Download source files for multiple papers in parallel."""
    if max_papers is not None:
        papers = papers[:max_papers]

    if num_threads is None:
        num_threads = max(1, os.cpu_count() // 2)

    manager = Manager()
    shared_stats = {
        'lock': manager.Lock(),
        'current': manager.Value('i', 0),
        'total': manager.Value('i', len(papers)),
        'successful': manager.Value('i', 0),
        'failed': manager.Value('i', 0)
    }

    client = ArxivClient()
    worker_func = partial(download_paper_worker,
                          output_dir=output_dir,
                          shared_stats=shared_stats,
                          client=client,
                          delay_seconds=client.delay_seconds)

    with Pool(processes=num_threads) as pool:
        results = pool.map(worker_func, papers)

    results_dict = {r['paper_id']: {
        'success': r['success'],
        'result': r['result'],
        'title': r['title']
    } for r in results}

    return results_dict


def main_download_cs_parallel(date_start, date_end, max_results, num_threads=None):
    client = ArxivClient()
    print(f"Searching for CS papers with source files from {date_start} to {date_end}...")

    papers = client.search_cs_papers_with_source(
        max_results=max_results,
        date_start=date_start,
        date_end=date_end
    )

    if papers:
        print(f"\nFound {len(papers)} papers with source files")
        results = parallel_download_sources(
            papers,
            "./arxiv_sources",
            num_threads=num_threads
        )

        print("\nDownload Results:")
        successful = sum(1 for r in results.values() if r['success'])
        failed = sum(1 for r in results.values() if not r['success'])

        print(f"\nSummary:")
        print(f"Total papers processed: {len(results)}")
        print(f"Successfully downloaded: {successful}")
        print(f"Failed downloads: {failed}")

        print("\nDetailed Results:")
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
    parser.add_argument("--max_results", type=int, default=100, help="Maximum number of results to return")
    parser.add_argument("--threads", type=int, help="Number of parallel download threads")

    args = parser.parse_args()
    main_download_cs_parallel(
        args.date_start,
        args.date_end,
        args.max_results,
        args.threads
    )
