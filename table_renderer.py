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
    # png_path: Optional[str]
    pdf_path: Optional[str]
    render_success: bool
    error_message: Optional[str] = None

class TableRenderer:
    def __init__(self):
        # Set up logging first
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('TableRenderer')
        
        self.latex_header = r'''\documentclass{standalone}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsbsy}
\usepackage{amsthm}
\begin{document}'''

        self.latex_footer = r'''\end{document}'''
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

    def render_table_to_pdf(self, table_content, output_path):
        try:
            output_dir = Path(output_path).parent
            table_name = Path(output_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tex_file = Path(tmp_dir) / "table.tex"
                pdf_file = Path(tmp_dir) / table_name
                latex_content = f"{self.latex_header}\n{table_content}\n{self.latex_footer}"
                tex_file.write_text(latex_content)
                
                try:
                    result = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", tex_file.name],
                        cwd=tmp_dir,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except subprocess.TimeoutExpired:
                    self.logger.error("pdflatex timed out")
                    return False, "pdflatex timed out"
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"pdflatex failed with code {e.returncode}")
                    self.logger.error(f"pdflatex stderr:\n{e.stderr}")
                    return False, f"pdflatex failed with code {e.returncode}"
                
                if result.returncode != 0:
                    self.logger.error(f"pdflatex output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                    return False, f"pdflatex failed: {result.stdout}\n{result.stderr}"
                
                if not pdf_file.exists():
                    self.logger.error("pdflatex did not produce a PDF file")
                    return False, "pdflatex did not produce a PDF file"
                
                # Move the generated PDF to the destination directory with the relevant name
                shutil.move(str(pdf_file), str(output_dir))
                
                self.logger.info(f"PDF successfully generated and moved to {output_dir}")
                return True, None
                
        except Exception as e:
            self.logger.error(f"Error in render_table_to_pdf: {str(e)}", exc_info=True)
            return False, str(e)
    
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
            
        # success, error_msg = self.render_to_png(cleaned_content, output_path)
        success, error_msg = self.render_table_to_pdf(cleaned_content, output_path)
        
        if success:
            pass
            # self.logger.info("Successfully rendered table")
        else:
            self.logger.error(f"Failed to render table: {error_msg}")
        
        return RenderedTable(
            cleaned_content=cleaned_content,
            pdf_path=output_path if success else None,
            render_success=success,
            error_message=error_msg
        )