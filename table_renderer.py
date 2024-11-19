# table_renderer.py

import re
import subprocess
import shutil
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Optional
import logging
import os

@dataclass
class RenderedTable:
    cleaned_content: str
    png_path: Optional[str]
    render_success: bool
    error_message: Optional[str] = None

class TableRenderer:
    def __init__(self):
        # Set up logging first
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('TableRenderer')
        
        self.latex_template = r"""
\documentclass{article}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{colortbl}
\usepackage{graphicx}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tabular}

\begin{document}
%s
\end{document}
"""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        missing_deps = []
        
        # Check for pdflatex
        pdflatex_path = shutil.which('pdflatex')
        if not pdflatex_path:
            missing_deps.append('pdflatex not found in PATH')
        else:
            pass
            # self.logger.debug(f"Found pdflatex at: {pdflatex_path}")
        
        # Check for magick
        magick_path = shutil.which('magick')
        if not magick_path:
            missing_deps.append('ImageMagick (magick) not found in PATH')
        else:
            pass
            # self.logger.debug(f"Found magick at: {magick_path}")
        
        if missing_deps:
            error_msg = "Missing required dependencies:\n" + "\n".join(missing_deps)
            raise RuntimeError(error_msg)

    def clean_content(self, table_content: str) -> str:
        """Clean table content by removing captions, comments, labels."""
        content = re.sub(r'(?<!\\)%.*?$', '', table_content, flags=re.MULTILINE)
        tabular_match = re.search(r'(\\begin{tabular}.*?\\end{tabular})', content, re.DOTALL)
        if not tabular_match:
            self.logger.warning("No tabular environment found in content")
            return ""
        
        tabular_content = tabular_match.group(1)
        tabular_content = re.sub(r'\\label{.*?}', '', tabular_content)
        tabular_content = re.sub(r'\\ref{.*?}', '', tabular_content)
        
        # self.logger.debug(f"Cleaned table content: {tabular_content}")
        return tabular_content.strip()

    def render_to_png(self, table_content: str, output_path: str) -> tuple[bool, Optional[str]]:
        """Render LaTeX table to PNG using minimal LaTeX document."""
        # Convert to absolute path
        output_path = os.path.abspath(output_path)
        # self.logger.debug(f"Starting render_to_png for output: {output_path}")
        
        # Ensure output directory exists with proper permissions
        output_dir = Path(output_path).parent
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Ensure directory is writable
            os.chmod(output_dir, 0o755)
            # self.logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            error_msg = f"Failed to create output directory: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
        
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tex_file = tmp_path / "table.tex"
                pdf_file = tmp_path / "table.pdf"
                temp_png = tmp_path / "table.png"
                
                # Create complete LaTeX document
                complete_doc = self.latex_template % table_content
                tex_file.write_text(complete_doc)
                
                # self.logger.debug(f"Created tex file at: {tex_file}")
                
                # Run pdflatex
                # self.logger.debug("Running pdflatex...")
                process = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_file.name],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if not pdf_file.exists():
                    error_msg = "PDF file was not created"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Fixed ImageMagick command
                # self.logger.debug("Running magick with adjusted quality settings...")
                process = subprocess.run(
                    [
                        "magick", "convert",  # Add convert command explicitly
                        str(pdf_file),        # Input file first
                        "-density", "150",
                        "-trim",              # Remove extra white space
                        "+repage",            # Reset page size
                        "-background", "white",
                        "-flatten",           # Flatten image
                        "-quality", "100",    # High quality
                        str(temp_png)         # Output file last
                    ],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if not temp_png.exists():
                    error_msg = "Temporary PNG file was not created"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # Copy the PNG to final destination
                # self.logger.debug(f"Copying PNG to final destination: {output_path}")
                shutil.copy2(temp_png, output_path)
                
                # Verify the output file was created
                if not Path(output_path).exists():
                    error_msg = f"Output file {output_path} was not created"
                    self.logger.error(error_msg)
                    return False, error_msg
                
                # self.logger.debug("Successfully created PNG file")
                return True, None
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed:\nStdout: {e.stdout}\nStderr: {e.stderr}"
            self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def process_table(self, table_content: str, output_path: str) -> RenderedTable:
        """Process table content and render to PNG."""
        # self.logger.info(f"Processing table for output: {output_path}")
        
        cleaned_content = self.clean_content(table_content)
        if not cleaned_content:
            error_msg = "No tabular environment found"
            self.logger.warning(error_msg)
            return RenderedTable(
                cleaned_content=cleaned_content,
                png_path=None,
                render_success=False,
                error_message=error_msg
            )
            
        success, error_msg = self.render_to_png(cleaned_content, output_path)
        
        if success:
            pass
            # self.logger.info("Successfully rendered table")
        else:
            self.logger.error(f"Failed to render table: {error_msg}")
        
        return RenderedTable(
            cleaned_content=cleaned_content,
            png_path=output_path if success else None,
            render_success=success,
            error_message=error_msg
        )