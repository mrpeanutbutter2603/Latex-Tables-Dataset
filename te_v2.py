import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import pandas as pd

@dataclass
class Table:
    content: str
    caption: Optional[str]
    section: Optional[str]
    subsection: Optional[str]
    paper_id: str
    paper_title: str
    source_file: str
    max_cols: int
    max_rows: int
    cell_types: Set[str]

def clean_tex_comments(content: str) -> str:
    """
    Clean LaTeX content by removing comments and extra whitespace.
    Handles both line comments and inline comments.
    """
    # First handle line continuations (lines ending with %)
    content = re.sub(r'%\s*\n\s*', ' ', content)
    
    # Remove line comments (% to end of line, but not \% which is literal %)
    content = re.sub(r'(?<!\\)%.*?$', '', content, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    content = re.sub(r'\s+', ' ', content)
    
    return content.strip()

class TexParser:
    def __init__(self, directory_path: str):
        self.directory = Path(directory_path)
        self.paper_id = self.directory.name  # Arxiv directory name as paper ID
        self.main_file_content = ""
        self.processed_files = set()
        self.section_positions = []
        self.latex_commands = {}
        
    def find_main_tex_file(self) -> Optional[Path]:
        """Find the main tex file in the directory."""
        for tex_path in self.directory.rglob('*.tex'):
            try:
                with open(tex_path, 'r', encoding='utf-8') as file:
                    content = file.read()
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
        # Clean the content first (remove comments, extra whitespace)
        cleaned_content = clean_tex_comments(self.main_file_content)
        
        # Extract title
        title_match = re.search(r'\\[a-zA-Z]*title\s*{', cleaned_content)
        if title_match:
            start_pos = title_match.end() - 1  # include the opening brace
            paper_title = self.post_process_latex_content(cleaned_content[start_pos:])
        else:
            paper_title = "Unknown Title"
            
        return paper_title

    def extract_latex_commands(self, content: str) -> dict:
        """Extract LaTeX command definitions into a dictionary."""
        commands = {}
        # Match commands with optional arguments and handle no-argument commands
        command_pattern = r'\\(?:newcommand|def|renewcommand){\\([^}]+)}(?:\[[\d]*\])?\s*{([^}]+)}'
        
        for match in re.finditer(command_pattern, content):
            command_name = match.group(1)
            command_value = match.group(2)
            commands[command_name] = self.post_process_latex_content(command_value)
        
        return commands

    def build_section_cache(self):
        """
        Build cache of section positions in a single pass through main_file_content.
        Stores tuples of (position, type, name) sorted by position.
        """
        # Clear any existing positions
        self.section_positions = []
        
        # Look for sections and subsections
        matched_sections = re.finditer(r'\\(section|subsection)\s*{', self.main_file_content)
        for match in matched_sections:
            position = match.start()
            section_type = match.group(1)  # Will be either "section" or "subsection"
            section_name = self.post_process_latex_content(self.main_file_content[match.end() - 1:])
            self.section_positions.append((position, section_type, section_name))
        
        # Sort by position
        # print(self.section_positions)
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

    def extract_caption(self, table_content: str) -> Optional[str]:
        """
        Extract and clean caption from table environment.
        Handles nested braces and removes comments.
        """
        # First clean any comments from the entire table content
        cleaned_content = clean_tex_comments(table_content)
        
        # Find caption command
        caption_match = re.search(r'\\caption\s*{', cleaned_content)
        if not caption_match:
            return None
        
        # Extract content within caption braces, handling nested braces
        start_pos = caption_match.end() - 1  # include the opening brace
        return self.post_process_latex_content(cleaned_content[start_pos:])

    def analyze_table_content(self, table_content: str):
        """Analyze table content to extract structural and content information."""
        # Extract the tabular environment
        tabular_match = re.search(r'\\begin{tabular}.*?\\end{tabular}', table_content, re.DOTALL)
        if not tabular_match:
            return None, None, None
        
        tabular_content = tabular_match.group(0)
        
        # Get row data
        rows = re.split(r'\\\\', tabular_content)
        rows = [row.strip() for row in rows if row.strip()]
        
        max_rows = len(rows)
        max_cols = 0
        cell_types = set()
        
        for row in rows:
            # Split by & to get cells
            cells = re.split(r'&', row)
            cells = [self.clean_cell_content(cell.strip()) for cell in cells if cell.strip()]
            max_cols = max(max_cols, len(cells))
            
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

    def extract_tables(self) -> List[Table]:
        """Extract all tables from the expanded content."""
        # Build section cache once
        self.build_section_cache()
        
        tables = []
        paper_title = self.extract_metadata()
        
        # Find all table environments
        table_pattern = r'\\begin{table}.*?\\end{table}'
        
        for table_match in re.finditer(table_pattern, self.main_file_content, re.DOTALL):
            table_content = table_match.group(0)
            table_pos = table_match.start()
            
            # Extract and clean caption
            caption = self.extract_caption(table_content)

            # Analyze table content
            max_cols, max_rows, cell_types = self.analyze_table_content(table_content)
            
            # Find current section using cached positions
            section, subsection = self.find_section_for_position(table_pos)
            
            # Find source file from markers
            source_file = "main.tex"
            marker_matches = list(re.finditer(
                r'% BEGIN_INCLUDE_FROM: (.*?)\n',
                self.main_file_content[:table_pos]
            ))
            if marker_matches:
                source_file = marker_matches[-1].group(1)
            
            table = Table(
                content=table_content,
                caption=caption,
                section=section,
                subsection=subsection,
                paper_id=self.paper_id,
                paper_title=paper_title,
                source_file=source_file,
                max_cols=max_cols,
                max_rows=max_rows,
                cell_types=cell_types
            )
            tables.append(table)
        
        return tables

    def process(self) -> pd.DataFrame:
        """Main processing function."""
        # Find and read main file
        main_file = self.find_main_tex_file()
        if not main_file:
            raise Exception("No main tex file found!")
            
        # Read and expand main file
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.processed_files.add(main_file)
            content = clean_tex_comments(content)
            self.main_file_content = self.expand_includes(content, main_file)
            self.latex_commands = self.extract_latex_commands(self.main_file_content)
        
        # Extract tables
        tables = self.extract_tables()
        
        # Print table information
        for table in tables:
            print(f"table paper_id: {table.paper_id}")
            print(f"table paper_title: {table.paper_title}")
            print(f"table section: {table.section}")
            print(f"table subsection: {table.subsection}")
            print(f"table caption: {table.caption}")
            print(f"table source_file: {table.source_file}")
            print(f"table max_cols: {table.max_cols}")
            print(f"table max_rows: {table.max_rows}")
            print(f"table cell_types: {table.cell_types}")
            print("-" * 50)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'paper_id': table.paper_id,
                'paper_title': table.paper_title,
                'section': table.section,
                'subsection': table.subsection,
                'table_content': table.content,
                'caption': table.caption,
                'source_file': table.source_file,
                'max_cols': table.max_cols,
                'max_rows': table.max_rows,
                'cell_types': table.cell_types
            }
            for table in tables
        ])
        
        return df

def main():
    # Example usage
    directory_path = "arxiv_sources/2411.09473"  # Example directory with extracted
    parser = TexParser(directory_path)
    df = parser.process()
    
    # Save to CSV
    df.to_csv('extracted_tables.csv', index=False)
    print(f"Processed {len(df)} tables")
    print(f"Output saved to extracted_tables.csv")

if __name__ == "__main__":
    main()