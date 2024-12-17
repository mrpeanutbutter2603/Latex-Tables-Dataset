import os
import re
from pathlib import Path
from table_renderer import TableRenderer
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import requests
import xml.etree.ElementTree as ET

@dataclass
class Table:
    content: str
    caption: Optional[str]
    section: Optional[str]
    subsection: Optional[str]
    paper_id: str
    paper_title: str
    paper_categories: List[str]
    source_file: str
    max_cols: int
    max_rows: int
    cell_types: Set[str]
    label: Optional[str] = None
    references: Optional[set[tuple]] = None  # List of all references to this table

def clean_tex_comments(content: str) -> str:
    """
    Clean LaTeX content by removing comments and extra whitespace.
    Handles both line comments and inline comments.
    """
    
    # Remove line comments (% to end of line, but not \% which is literal %)
    content = re.sub(r'(?<!\\)%.*?$', '', content, flags=re.MULTILINE)

    # First handle line continuations (lines ending with %)
    content = re.sub(r'%\s*\n\s*', ' ', content)
    
    # Clean up extra whitespace
    content = re.sub(r'\s+', ' ', content)

    # Remove explicit comments i.e. content between \begin{comment} and \end{comment}
    content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)

    # Remove content between \iffalse and \fi
    content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
    
    return content.strip()

def query_arxiv_api(arxiv_id: str) -> Tuple[str, List[str]]:
    """Query the arXiv API to get the paper title and categories."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        entry = root.find('atom:entry', ns)
        if entry is not None:
            title = entry.find('atom:title', ns).text.strip()
            categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
            return title, categories
        else:
            print(f"No entry found for arXiv ID: {arxiv_id}")
            return "Unknown Title", []
    else:
        print(f"Error querying arXiv API: {response.status_code}")
        return "Unknown Title", []

class TexParser:
    def __init__(self, directory_path: str):
        self.directory = Path(directory_path)
        self.paper_id = self.directory.name  # Arxiv directory name as paper ID
        self.main_file_content = ""
        self.processed_files = set()
        self.section_positions = []
        self.latex_commands = {}

        # Create output directory with proper permissions
        self.output_dir = self.directory / "rendered_tables"
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.output_dir, 0o755)
        except Exception as e:
            print(f"Warning: Failed to create output directory: {e}")
        
        self.renderer = TableRenderer()
        self.label_refs = {}  # Dictionary to store all references: label -> List[RefInfo]
        self.total_papers = 0  # Total papers processed
        self.nested_tabular_papers = 0  # Count of papers with nested tabular environments

    def build_reference_dictionary(self):
        """
        Build a dictionary of all references in the document.
        Maps labels to lists of RefInfo objects containing reference information.
        """
        # Look for references to tables using various reference commands
        ref_pattern = r'\\(?:auto|c|C|tab)?ref{([^}]+)}'
        
        # Find all references
        for match in re.finditer(ref_pattern, self.main_file_content):
            label = match.group(1)
            ref_pos = match.start()
            
            # Get section information
            section, subsection = self.find_section_for_position(ref_pos)
            
            # Get context
            context = self.find_reference_context(match, self.main_file_content)
            ref_info = (section, subsection, context['prev'], context['current'], context['next'])
            
            # Add to dictionary
            if label not in self.label_refs:
                self.label_refs[label] = set()

            if ref_info not in self.label_refs[label]:
                self.label_refs[label].add(ref_info)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling LaTeX specifics."""
        # First, protect certain patterns
        protected_text = text
        
        # Protect LaTeX commands with periods
        protected_text = re.sub(r'\\[a-zA-Z]+\.[a-zA-Z]+', 
                              lambda m: m.group().replace('.', '@'), 
                              protected_text)
        
        # Protect common abbreviations
        abbreviations = ['e.g', 'i.e', 'et al', 'etc', 'vs', 'Fig', 'fig', 'Eq', 'eq',
                        'cf', 'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr']
        for abbr in abbreviations:
            protected_text = re.sub(rf'\b{abbr}\.',
                                  lambda m: m.group().replace('.', '@'),
                                  protected_text)
        
        # Protect decimal numbers
        protected_text = re.sub(r'\d+\.\d+',
                              lambda m: m.group().replace('.', '@'),
                              protected_text)
        
        # Split on sentence boundaries
        # Look for: period/exclamation/question mark + space + capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)
        
        # Restore protected periods and clean sentences
        sentences = [s.replace('@', '.').strip() for s in sentences]
        return [s for s in sentences if s]

    def find_reference_context(self, ref_match: re.Match, text: str) -> Dict[str, str]:
        """Find the surrounding context of a reference."""
        start_pos = ref_match.start()
        
        # Get the paragraph containing the reference
        # Look for double newlines or section commands to identify paragraph boundaries
        paragraph_pattern = r'(?:(?!\n\n|\\(?:sub)*section\*?).)+(?:\n(?!\n|\\(?:sub)*section\*?).+)*'
        paragraph_matches = list(re.finditer(paragraph_pattern, text))
        
        containing_paragraph = None
        for para in paragraph_matches:
            if para.start() <= start_pos <= para.end():
                containing_paragraph = para
                break
        
        if not containing_paragraph:
            return {"prev": "", "current": "", "next": ""}
        
        # Get paragraph text and split into sentences
        para_text = containing_paragraph.group()

        # Remove LaTeX environments
        environments = ['table', 'figure', 'minipage']
        for env in environments:
            # Remove \begin{env}...\end{env} and \begin{env*}...\end{env*}
            para_text = re.sub(rf'\\begin{{{env}\*?}}.*?\\end{{{env}\*?}}', '', para_text, flags=re.DOTALL)

         # Remove all sectioning commands like \section, \subsection, \subsubsection, etc.
        para_text = re.sub(r'\\(?:sub)*section\*?{[^}]*}', '', para_text)
        para_text = re.sub(r'(?:sub)*section\*?{[^}]*}', '', para_text)

        # Clean the paragraph text first
        para_text = clean_tex_comments(para_text)  # Remove LaTeX comments
        para_text = re.sub(r'\s+', ' ', para_text)  # Normalize whitespace
        
        sentences = self.split_into_sentences(para_text)
        
        # Find the sentence containing the reference
        ref_text = ref_match.group()
        current_sentence = ""
        current_idx = -1
        
        for idx, sentence in enumerate(sentences):
            if ref_text in sentence:
                current_sentence = sentence
                current_idx = idx
                break
        
        # Get previous and next sentences if they exist in the same paragraph
        prev_sentence = sentences[current_idx - 1] if current_idx > 0 else ""
        next_sentence = sentences[current_idx + 1] if current_idx < len(sentences) - 1 else ""
        

        # Clean the extracted sentences
        current_sentence = self.clean_latex(current_sentence)
        prev_sentence = self.clean_latex(prev_sentence)
        next_sentence = self.clean_latex(next_sentence)
        
        return {
            "prev": prev_sentence,
            "current": current_sentence,
            "next": next_sentence
        }

    def find_main_tex_file(self) -> Optional[Path]:
        """Find the main tex file in the directory."""
        for tex_path in self.directory.rglob('*.tex'):
            # print(f"Checking {tex_path}")
            try:
                with open(tex_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # print(content)
                    if '\\documentclass' in content and '\\begin{document}' in content:
                        return tex_path
            except Exception as e:
                print(f"Error reading {tex_path}: {e}")
        return None

    def resolve_input_path(self, input_name: str, current_file: Path) -> Optional[Path]:
        """Resolve the path of an included file relative to current file."""
        current_dir = current_file.parent
        
        # Try different possible paths
        possible_paths = [
            current_dir / f"{input_name}.tex",
            current_dir / input_name,
            current_dir / f"{input_name}.tex",
            self.directory / input_name,
            self.directory / f"{input_name}.tex"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        print(f"Warning: Could not resolve input file {input_name} relative to {current_file}")
        return None

    def expand_includes(self, content: str, current_file: Path) -> str:
        """Replace input and include commands with actual content."""
        def replace_include(match):
            input_name = match.group(1).strip()
            resolved_path = self.resolve_input_path(input_name, current_file)
            
            if resolved_path and resolved_path not in self.processed_files:
                try:
                    self.processed_files.add(resolved_path)
                    with open(resolved_path, 'r', encoding='utf-8') as f:
                        included_content = clean_tex_comments(f.read())
                        # Add a comment to mark the source file
                        marker = f"\n% BEGIN_INCLUDE_FROM: {resolved_path}\n"
                        end_marker = f"\n% END_INCLUDE_FROM: {resolved_path}\n"
                        marked_content = marker + included_content + end_marker
                        return self.expand_includes(marked_content, resolved_path)
                except Exception as e:
                    print(f"Error reading included file {resolved_path}: {e}")
                    return ""
            return ""

        # Handle both \input{file} and \include{file}
        expanded = re.sub(r'\\(?:input|include){([^}]+)}', replace_include, content)
        return expanded

    def extract_metadata(self) -> tuple[str, List[str]]:
        """Extract paper title and authors from the expanded content."""
        """
        # Clean the content first (remove comments, extra whitespace)
        cleaned_content = clean_tex_comments(self.main_file_content)
        
        # Extract title
        title_match = re.search(r'\\[a-zA-Z]*title(\[[^\]]*\])?\s*{', cleaned_content)
        if title_match:
            start_pos = title_match.end() - 1  # include the opening brace
            paper_title = self.post_process_latex_content(cleaned_content[start_pos:])
        else:
            paper_title = "Unknown Title"
        """

        paper_title, categories = query_arxiv_api(self.paper_id)
        return paper_title, categories

    def extract_latex_commands(self, content: str) -> dict:
        """Extract LaTeX command definitions into a dictionary."""
        commands = {}
        # Match commands with optional arguments and handle no-argument commands
        command_pattern = r'\\(?:newcommand|def|renewcommand){\\([^}]+)}(?:\[[\d]*\])?\s*{([^}]+)}'
        matches = re.findall(command_pattern, content)
        commands = {}
        for match in matches:
            command_name = match[0]
            command_value = match[1]
            command_value = re.sub(r'\\[a-zA-Z]+' , '', command_value)
            command_value = command_value.replace('{', '').replace('}', '')
            command_value = command_value.strip()
            commands[command_name] = command_value

        return commands

    def build_section_cache(self):
        """
        Build cache of section positions in a single pass through main_file_content.
        Stores tuples of (position, type, name) sorted by position.
        """
        # Clear any existing positions
        self.section_positions = []
        
        # Look for sections and subsections
        matched_sections = re.finditer(r'\\(section|subsection)\*?\s*{', self.main_file_content)
        for match in matched_sections:
            position = match.start()
            section_type = match.group(1)  # Will be either "section" or "subsection"
            section_name = self.post_process_latex_content(self.main_file_content[match.end() - 1:])
            self.section_positions.append((position, section_type, section_name))
        
        # Sort by position
        self.section_positions.sort()

    def extract_bracketed_content(self, content: str, depth: int = 1) -> str:
        """
        Extract content within nested curly braces, handling nested braces properly.
        depth parameter determines how many levels of nested braces to process.
        """
        result = []
        brace_count = 0
        current_char_idx = 0
        
        while current_char_idx < len(content):
            char = content[current_char_idx]
            
            if char == '{':
                brace_count += 1
                if brace_count == 1:
                    # Start of content we want to capture
                    current_char_idx += 1
                    continue
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
            
            if brace_count >= 1:
                result.append(char)
            
            current_char_idx += 1
        
        extracted = ''.join(result)
        
        # If we need to process more levels of nesting
        if depth > 1 and '{' in extracted:
            # Process each nested brace group
            parts = []
            last_end = 0
            for match in re.finditer(r'{([^{}]*)}', extracted):
                parts.append(extracted[last_end:match.start()])
                nested_content = self.extract_bracketed_content(match.group(1), depth - 1)
                parts.append(nested_content)
                last_end = match.end()
            parts.append(extracted[last_end:])
            extracted = ''.join(parts)
        
        return extracted

    def post_process_latex_content(self, content: str) -> str:
        content = self.extract_bracketed_content(content)
        content = content.strip()
        content = re.sub(r'\\label{[^}]+}', '', content)
        content = self.resolve_commands(content, self.latex_commands)
        return self.clean_latex(content)

    def clean_latex(self, text: str) -> str:
        """Remove LaTeX commands and cleanup text."""
        # Replace line breaks with space
        text = re.sub(r'\s*\\\\\s*', ' ', text)
        # Remove \thanks{...} content
        text = re.sub(r'\\thanks{[^}]*}', '', text)
        # Remove common LaTeX formatting commands
        text = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        # Remove dollar signs
        text = re.sub(r'\$', '', text)
        # Remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def find_section_for_position(self, position: int) -> tuple[str, str]:
        """
        Find the current section and subsection for a given position
        using the cached section positions.
        """
        current_section = "Unknown Section"
        current_subsection = "Unknown Section"
        
        # Find the latest section and subsection before this position
        for pos, type_, name in self.section_positions:
            if pos > position:
                break
                
            if type_ == 'section':
                current_section = name
                current_subsection = name  # Reset subsection at new section
            elif type_ == 'subsection':
                current_subsection = name
                
        return current_section, current_subsection
    
    def resolve_commands(self, text: str, commands: dict) -> str:
        """Replace command references with their values."""
        # Replace each command reference with its value
        for cmd, value in commands.items():
            text = text.replace(f"\\{cmd}", value)
        return text

    def extract_label(self, table_content: str) -> Optional[str]:
        """Extract label from table environment."""
        label_matches = list(re.finditer(r'\\label{([^}]+)}', table_content))
        if not label_matches:
            return None
        
        labels = [match.group(1) for match in label_matches]
        return labels

    def extract_caption(self, table_content: str) -> Optional[str]:
        """
        Extract and clean caption from table environment.
        Handles nested braces and removes comments.
        """
        # First clean any comments from the entire table content
        cleaned_content = clean_tex_comments(table_content)
        
        # Extract all captions
        caption_matches = list(re.finditer(r'\\caption\*?(?:of{[^}]+})?\s*{', cleaned_content))
        if not caption_matches:
            return None
        
        # Extract content within caption braces, handling nested braces
        cleaned_captions = []
        for caption_match in caption_matches:
            start_pos = caption_match.end() - 1  # include the opening brace
            cleaned_caption = self.post_process_latex_content(cleaned_content[start_pos:])
            cleaned_captions.append(cleaned_caption)

        return cleaned_captions

    def analyze_table_content(self, tabular_content: str):
        """Analyze table content to extract structural and content information."""

        # Find all \begin{tabular} and \end{tabular} positions
        begin_tabular_positions = [m.start() for m in re.finditer(r'\\begin{tabular\*?}', tabular_content)]
        end_tabular_positions = [m.end() for m in re.finditer(r'\\end{tabular\*?}', tabular_content)]

        # Keep the first \begin{tabular} and the last \end{tabular}
        first_begin = begin_tabular_positions[0]
        last_end = end_tabular_positions[-1]

        # Substitute all other \begin{tabular} and \end{tabular} with a placeholder
        placeholder = "TABLE_PLACEHOLDER"
        tabular_content = (
            tabular_content[:first_begin + len(r'\begin{tabular}')] +
            re.sub(r'\\begin{tabular\*?}.*?\\end{tabular\*?}', placeholder, tabular_content[first_begin + len(r'\begin{tabular}'):last_end], flags=re.DOTALL) +
            tabular_content[last_end:]
        )

        # Split into rows by main row delimiters
        rows = re.split(r'\\\\', tabular_content)  # Split LaTeX rows ending with \\
        rows = [row.strip() for row in rows if row.strip()]  # Remove empty rows

        max_rows = max(len(rows) - 1, 1)
        max_cols = 0
        cell_types = set()

        for row in rows:
            # Split by & to get cells
            cells = re.split(r'&', row)  # Cells are separated by &
            cells = [self.clean_cell_content(cell.strip()) for cell in cells if cell.strip()]
            max_cols = max(max_cols, len(cells))  # Update maximum columns in any row

            # Analyze cell types
            for cell in cells:
                cell_type = self.determine_cell_type(cell)
                cell_types.add(cell_type)

        return max_cols, max_rows, cell_types

    def clean_cell_content(self, cell: str) -> str:
        """Clean LaTeX formatting from cell content."""
        # Remove common LaTeX commands
        cell = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', cell)
        cell = re.sub(r'\\[a-zA-Z]+', '', cell)
        return cell.strip()

    def determine_cell_type(self, cell: str) -> str:
        """
        Determine the type of content in a table cell.
        Returns a descriptive string of the content type.
        """
        # Clean the cell content first
        cell = cell.strip()
            
        if not cell:
            return 'empty'
        
        # Check for TABLE_PLACEHOLDER
        if "TABLE_PLACEHOLDER" in cell:
            return 'table'

        # Check for numeric types with units
        if re.match(r'^-?\d+\.?\d*\s*[a-zA-Z%]+$', cell):
            return 'measurement'
            
        # Check for percentages
        if re.match(r'^-?\d+\.?\d*\s*%$', cell):
            return 'percentage'
            
        # Check for currency
        if re.match(r'^[$€£¥]\s*-?\d+\.?\d*$|^-?\d+\.?\d*\s*[$€£¥]$', cell):
            return 'currency'
            
        # Check for ranges
        if re.match(r'^\d+\s*-\s*\d+$|^\d+\s*to\s*\d+$', cell):
            return 'range'
            
        # Check for dates
        if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', cell):
            return 'date'
            
        # Check for times
        if re.match(r'^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?$', cell):
            return 'time'
            
        # Check for equations/mathematical expressions
        if any(x in cell for x in ['+', '-', '*', '/', '=', '±', '×', '÷', '∑', '∏']):
            return 'mathematical'
            
        # Check for boolean values
        if cell.lower() in ['true', 'false', 'yes', 'no', '✓', '✗', '×']:
            return 'boolean'

        # Check for integers
        try:
            int(cell)
            return 'integer'
        except ValueError:
            # Check for floating point numbers
            try:
                float(cell)
                return 'decimal'
            except ValueError:
                # If nothing else matches, check if it's a multi-word text
                if ' ' in cell:
                    return 'text'
                return 'categorical'

    def extract_tabular_content(self, content: str) -> list:
        """
        Extract all tabular environments, including nested ones.
        """
        tabular_stack = []
        tabular_pattern = re.compile(r'\\begin{tabular\*?}|\\end{tabular\*?}')
        tabular_contents = []
        current_content = []
        for match in tabular_pattern.finditer(content):
            if re.match(r'\\begin{tabular\*?}', match.group()):
                if tabular_stack:
                    current_content.append(match.group())
                tabular_stack.append(match.start())
            elif re.match(r'\\end{tabular\*?}', match.group()):
                if not tabular_stack:
                    continue  # Unmatched \end{tabular}
                start_pos = tabular_stack.pop()
                if not tabular_stack:
                    # Top-level tabular environment
                    tabular_contents.append(content[start_pos:match.end()])
                else:
                    current_content.append(match.group())
            if tabular_stack:
                current_content.append(content[match.start():match.end()])
        return tabular_contents

    def has_nested_tabular(self, content: str) -> bool:
        """
        Check if the content has nested tabular environments.
        """
        tabular_stack = []
        tabular_pattern = re.compile(r'\\begin{tabular\*?}|\\end{tabular\*?}')
        for match in tabular_pattern.finditer(content):
            if re.match(r'\\begin{tabular\*?}', match.group()):
                tabular_stack.append(match.start())
            elif re.match(r'\\end{tabular\*?}', match.group()):
                if not tabular_stack:
                    return False  # Unmatched \end{tabular}
                tabular_stack.pop()
            if len(tabular_stack) > 1:
                return True  # Nested \begin{tabular} found
        return False

    def extract_tables(self) -> List[Table]:
        """Extract all tables from the expanded content."""
        # Build section cache once
        self.build_section_cache()
        
        tables = []
        paper_title, categories = self.extract_metadata()
        
        outer_match_pattern = r'\\begin{(?:table|figure)\*?}\s*(.*?)\\end{(?:table|figure)\*?}'
        outer_matches = list(re.finditer(outer_match_pattern, self.main_file_content, re.DOTALL))
        if not outer_matches:
            return tables
        
        for outer_match in outer_matches:
            outer_content = outer_match.group(0)
            outer_content_pos = outer_match.start()

            ''''
            # Find all table environments
            table_pattern = r'(\\begin{tabular}.*?\\end{tabular})'
            tabular_matches = list(re.finditer(table_pattern, outer_content, re.DOTALL))
            if not tabular_matches:
                continue
            '''

            # Extract all tabular environments
            tabular_contents = self.extract_tabular_content(outer_content)
            if not tabular_contents:
                continue
                
            # Check for nested tabular environments
            if self.has_nested_tabular(outer_content):
                print(f"Warning: Nested tabular environments found in {paper_title}")
                if self.nested_tabular_papers == 0:
                    self.nested_tabular_papers += 1
            #    return pd.DataFrame()
        
            # Find source file from markers
            source_file = "main.tex"
            marker_matches = list(re.finditer(
                r'% BEGIN_INCLUDE_FROM: (.*?)\n',
                self.main_file_content[:outer_content_pos]
            ))

            if marker_matches:
                source_file = marker_matches[-1].group(1)

            # Find current section using cached positions
            section, subsection = self.find_section_for_position(outer_content_pos)
            for table_content in tabular_contents:
                table_end = outer_content.find(table_content) + len(table_content)

                # Extract and clean caption
                # First check if the caption is before the tabular content
                caption = self.extract_caption(outer_content[:table_end])
                if caption:
                    caption = caption[-1]  # Use the last caption if multiple found
                else:
                    # If not, check if the caption is after the tabular content
                    caption = self.extract_caption(outer_content[table_end:])
                    if caption:
                        caption = caption[0]
                    else:
                        caption = None

 
                # Extract label
                # First check if the label is before the tabular content
                label = self.extract_label(outer_content[:table_end])
                if label:
                    label = label[-1]
                else:
                    # If not, check if the label is after the tabular content
                    label = self.extract_label(outer_content[table_end:])
                    if label:
                        label = label[0]
                    else:
                        label = None

                # Build reference dictionary if not already built
                if not self.label_refs:
                    self.build_reference_dictionary()

                # Find references to this table
                references = list(self.label_refs.get(label, []) if label else [])

                # Analyze table content
                max_cols, max_rows, cell_types = self.analyze_table_content(table_content)
                
                table = Table(
                    content=table_content,
                    caption=caption,
                    section=section,
                    subsection=subsection,
                    paper_id=self.paper_id,
                    paper_title=paper_title,
                    paper_categories=categories,
                    source_file=source_file,
                    max_cols=max_cols,
                    max_rows=max_rows,
                    cell_types=cell_types,
                    label=label,
                    references=references
                )
            
                tables.append(table)
        
        return tables

    def process(self) -> pd.DataFrame:
        """Main processing function."""
        # Find and read main file
        main_file = self.find_main_tex_file()
        if not main_file:
            # raise Exception("No main tex file found!")
            print("No main tex file found!")
            return pd.DataFrame()
            
        self.total_papers += 1
        # Read and expand main file
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.processed_files.add(main_file)
            content = clean_tex_comments(content)
            self.main_file_content = self.expand_includes(content, main_file)
            self.latex_commands = self.extract_latex_commands(self.main_file_content)
        
        # Extract tables
        tables = self.extract_tables()

        df_data = []
        for i, table in enumerate(tables):
            table_data = {
                'paper_id': table.paper_id,
                'paper_title': table.paper_title,
                'paper_categories': table.paper_categories,
                'section': table.section,
                'subsection': table.subsection,
                'table_content': table.content,
                'caption': table.caption,
                'max_cols': table.max_cols,
                'max_rows': table.max_rows,
                'cell_types': table.cell_types,
                'context': []
            }

            table_pdf_path = self.output_dir / f"{table.paper_id}_{i}.pdf"
            self.renderer.process_table(table.content, table_pdf_path)

            # Add reference information
            if table.references:
                context_set = set()
                for ref in table.references:
                    curr_context = f"{ref[2]} {ref[3]} {ref[4]}"
                    context_set.add(curr_context)

                context_list = list(context_set)

                # Add all references as lists
                table_data.update({
                    'context': context_list
                })

                # Print the sentences for debugging
                """
                print("Reference sentences:")
                for ref in table.references:
                    print(f"Section: {ref[0]}")
                    print(f"Subsection: {ref[1]}")
                    print(f"Prev: {ref[2]}")
                    print(f"Current: {ref[3]}")  # This should be the reference itself
                    print(f"Next: {ref[4]}")
                print("=" * 50)
                """

            print(f"table paper_id: {table.paper_id}")
            print(f"table paper_title: {table.paper_title}")
            print(f"table paper_categories: {table.paper_categories}")
            print(f"table section: {table.section}")
            print(f"table subsection: {table.subsection}")
            print(f"table caption: {table.caption}")
            print(f"table source_file: {table.source_file}")
            print(f"table max_cols: {table.max_cols}")
            print(f"table max_rows: {table.max_rows}")
            print(f"table cell_types: {table.cell_types}")
            print(f"table label: {table.label}")
            print(f"table context:")
            for i, context in enumerate(table_data['context']):
                print(f"{i + 1} --> {context}")
            print("-" * 50)
            df_data.append(table_data)
        
        return pd.DataFrame(df_data)

def main_single(paper):
    # Example usage
    directory_path = f"arxiv_sources/{paper}"  # Example directory with extracted
    subd = directory_path.split("/")[-1]
    parser = TexParser(directory_path)
    df = parser.process()
    OUTPUT_CSV_DIR = "output_csv_files"
    if not os.path.exists(OUTPUT_CSV_DIR):
        os.makedirs(OUTPUT_CSV_DIR)
    output_file = os.path.join(OUTPUT_CSV_DIR, f'{subd}_extracted_tables.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed {len(df)} tables")
    print(f"Output saved to {subd}_extracted_tables.csv")
    if parser.nested_tabular_papers > 0:
        print(f"Nested tabular environment found!")

def process_paper(paper_dir, base_dir, shared_stats):
    """Process a single paper directory."""
    try:
        directory_path = os.path.join(base_dir, paper_dir)
        parser = TexParser(directory_path)
        df = parser.process()
        
        # Update shared statistics
        if not df.empty:
            with shared_stats['lock']:
                shared_stats['processed_papers'].value += parser.total_papers
                shared_stats['nested_papers'].value += parser.nested_tabular_papers
        
        # Save individual paper results
        OUTPUT_CSV_DIR = "output_csv_files"
        if not os.path.exists(OUTPUT_CSV_DIR):
            os.makedirs(OUTPUT_CSV_DIR)
        output_file = os.path.join(OUTPUT_CSV_DIR, f'{paper_dir}_extracted_tables.csv')
        df.to_csv(output_file, index=False)
        
        return df if not df.empty else None
        
    except Exception as e:
        print(f"Error processing {paper_dir}: {e}")
        return None

def main(num_threads=None):
    """Main processing function with parallel execution."""
    if num_threads is None:
        num_threads = cpu_count()  # Use all available CPU cores by default
    
    directory_path = "arxiv_sources"
    subdirectories = [d for d in os.listdir(directory_path) 
                     if os.path.isdir(os.path.join(directory_path, d))]
    
    # Set up shared statistics using Manager
    manager = Manager()
    shared_stats = {
        'processed_papers': manager.Value('i', 0),
        'nested_papers': manager.Value('i', 0),
        'lock': manager.Lock()
    }
    
    # Create partial function with fixed arguments
    process_func = partial(process_paper, 
                         base_dir=directory_path, 
                         shared_stats=shared_stats)
    
    # Process papers in parallel
    with Pool(processes=num_threads) as pool:
        all_dfs = pool.map(process_func, subdirectories)
    
    # Filter out None results and combine DataFrames
    valid_dfs = [df for df in all_dfs if df is not None]
    if valid_dfs:
        final_df = pd.concat(valid_dfs, axis=0)
        
        # Save combined results
        output_file = os.path.join("output_csv_files", 'all_extracted_tables.xlsx')
        final_df.to_excel(output_file, index=False)
        
        # Print statistics
        empty_dfs = len(subdirectories) - len(valid_dfs)
        percent_nested = ((shared_stats['nested_papers'].value / 
                          shared_stats['processed_papers'].value) * 100 
                         if shared_stats['processed_papers'].value > 0 else 0)
        
        print(f"Output saved to {output_file}")
        print(f"Processed {len(subdirectories)} directories, {empty_dfs} had no tables")
        print(f"Total papers processed: {shared_stats['processed_papers'].value}")
        print(f"Papers with nested tabular environments: "
              f"{shared_stats['nested_papers'].value} ({percent_nested:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", 
                       help="Run for a single directory")
    parser.add_argument("--paper", type=str, 
                       help="Run for a single paper")
    parser.add_argument("--threads", type=int, 
                       help="Number of parallel threads to use")
    args = parser.parse_args()
    
    if args.single:
        if not args.paper:
            print("Please provide the paper name")
            exit(1)
        main_single(args.paper)
    else:
        main(num_threads=args.threads)




""""
def main():
    # Example usage
    directory_path = "arxiv_sources"  # Example directory with extracted
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    all_dfs = []
    empty_dfs = 0
    total_papers = 0
    total_nested_tabular_papers = 0
    for subd in subdirectories:
        print(f"Processing {subd}...")
        parser = TexParser(os.path.join(directory_path, subd))
        df = parser.process()
        if not df.empty:
            all_dfs.append(df)
        else:
            empty_dfs += 1
        OUTPUT_CSV_DIR = "output_csv_files"
        if not os.path.exists(OUTPUT_CSV_DIR):
            os.makedirs(OUTPUT_CSV_DIR)
        output_file = os.path.join(OUTPUT_CSV_DIR, f'{subd}_extracted_tables.csv')
        df.to_csv(output_file, index=False)
        total_papers += parser.total_papers
        total_nested_tabular_papers += parser.nested_tabular_papers
        print(f"Processed {len(df)} tables")
        print(f"Output saved to {subd}_extracted_tables.csv")
        print("=" * 50)

    final_df = pd.concat(all_dfs, axis=0)
    output_file = os.path.join(OUTPUT_CSV_DIR, f'all_extracted_tables.xlsx')
    final_df.to_excel(output_file, index=False)
    print(f"Output saved to {output_file}")
    print(f"Processed {len(subdirectories)}, {empty_dfs} directories had no tables")
    percent_nested_tabular = (total_nested_tabular_papers / total_papers) * 100 if total_papers > 0 else 0
    print(f"Total papers processed: {total_papers}")
    print(f"Total papers with nested tabular environments: {total_nested_tabular_papers} ({percent_nested_tabular:.2f}%)")

if __name__ == "__main__":
    # Whether we want to run a single instance or all should be decide from the command
    # line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Run for a single directory")
    parser.add_argument("--paper", type=str, help="Run for a single paper")
    args = parser.parse_args()
    if args.single:
        # Make sure the paper name is provided
        if not args.paper:
            print("Please provide the paper name")
            exit(1)

        main_single(args.paper)
    else:
        main()
"""