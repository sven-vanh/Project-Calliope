# PDFGenerator
# ========
# A simple wrapper for creating PDF reports with consistent styles.
# Supports adding headers, paragraphs, matplotlib figures, pandas DataFrames, and saving to PDF.
# Created By: Sven van Helten
# Date: 17.04.2025
# Version: 1.0
# License: MIT

# Import dependencies
import base64
import logging
import re
from io import BytesIO
from typing import List, Optional

import pandas as pd
from matplotlib.figure import Figure
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFGenerator:
    """
    A simple wrapper for creating PDF reports with consistent styles.
    Supports adding headers, paragraphs, matplotlib figures, pandas DataFrames, and saving to PDF.
    """
    def __init__(
        self, 
        filename: str, 
        api_key: str, 
        pagesize=letter, 
        api_base_url: Optional[str] = None, 
        model: str = "gpt-4o", 
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize the PDF document, set up base styles and OpenAI client.

        Args:
            filename: Output PDF file path
            api_key: OpenAI API key for generating text descriptions
            pagesize: ReportLab pagesize (default: letter)
            api_base_url: OpenAI API base URL (optional)
            model: OpenAI model to use for text generation (default: "gpt-4o")
            system_prompt: The system prompt to pass to the LLM (optional)
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Configure PDF parameters
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=pagesize)
        self.styles = getSampleStyleSheet()

        # Configure document styles
        self.base_paragraph_style = ParagraphStyle(
            'Base',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            spaceAfter=8,
        )
        self.header_style = ParagraphStyle(
            'Header',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            spaceAfter=12,
            textColor=colors.darkblue
        )

        # Configure OpenAI client
        self.openai_client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.model = model
        self.system_prompt = system_prompt
        
        # Initialize document elements
        self.elements = []
        self.logger.info(f"PDFGenerator initialized with filename: {filename}")


    def add_header(self, text: str) -> None:
        """
        Add a header to the document.

        Args:
            text: Header text content
        """
        # Add header with consistent styling
        self.elements.append(Spacer(1, 0.15*inch))
        self.elements.append(Paragraph(text, self.header_style))
        self.logger.info("Added header")


    def add_paragraph(self, text: str) -> None:
        """
        Add a paragraph of text to the document.

        Args:
            text: Paragraph text content
        """
        self.elements.append(Paragraph(text, self.base_paragraph_style))
        self.logger.info("Added paragraph")


    def add_figure(self, fig: Figure, width: float = 5.5, height: float = 3.5) -> None:
        """
        Add a matplotlib Figure object to the document.

        Args:
            fig: Matplotlib figure object
            width: Width in inches (default: 5.5)
            height: Height in inches (default: 3.5)
        """
        # Create in-memory buffer for the image and convert to PNG
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Create an Image object and add it to the document
        img = Image(buf, width*inch, height*inch)
        self.elements.append(img)
        self.elements.append(Spacer(1, 0.15*inch))
        self.logger.info("Added figure")


    def add_dataframe(self, df: pd.DataFrame, max_rows: Optional[int] = 20) -> None:
        """
        Add a pandas DataFrame as a styled table to the document.

        Args:
            df: Pandas DataFrame to be included as a table
            max_rows: Maximum number of rows to include (default: 20)
        """
        # Truncate large tables if necessary
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows)

        # Convert DataFrame to list format required by ReportLab
        columns: list = df.columns.tolist() if isinstance(df.columns.tolist(), list) else [df.columns.tolist()]
        values: list = df.values.tolist() if isinstance(df.values.tolist(), list) else [df.values.tolist()]
        data: list = columns + values
        table = Table(data, hAlign='LEFT')

        # Apply consistent, professional styling to the table
        style = TableStyle([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),

            # Data rows styling
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),

            # Grid and alternating row colors
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ])
        table.setStyle(style)

        # Add the table to the document
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.15*inch))
        self.logger.info("Added DataFrame as table")


    def add_page_break(self) -> None:
        """
        Add a page break to the document.
        """
        self.elements.append(PageBreak())
        self.logger.info("Added page break")


    def add_markdown(self, markdown_text: str) -> None:
        """
        Convert markdown text to ReportLab elements and add to the document.
        
        Args:
            markdown_text: Text in markdown format
        """
        elements = self._markdown_to_reportlab_elements(markdown_text)
        self.elements.extend(elements)
        self.logger.info("Added markdown text")


    def save(self) -> None:
        """
        Build and save the PDF document.
        """
        self.doc.build(self.elements)
        self.logger.info(f"PDF document saved as {self.filename}")


    def add_paragraph_from_figure(
        self, 
        fig: Figure, 
        prompt: str,
        width: float = 5.5, 
        height: float = 3.5
    ) -> None:
        """
        Add a matplotlib figure to the document with AI-generated explanatory text.

        Args:
            fig: Matplotlib figure object
            prompt: Instruction for OpenAI to generate text about the figure
            width: Width in inches (default: 5.5)
            height: Height in inches (default: 3.5)
        """
        # First add the figure as normal
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width*inch, height*inch)
        self.elements.append(img)
        self.elements.append(Spacer(1, 0.15*inch))

        # Convert figure to base64 for API call
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')

        # Call OpenAI API to generate text about the image
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Extract the generated text
        generated_text = response.choices[0].message.content

        # Add a subheader for the AI analysis
        analysis_header = Paragraph("Analysis:", ParagraphStyle(
            'AnalysisHeader',
            parent=self.styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.darkblue
        ))
        self.elements.append(analysis_header)

        # Add the generated text as markdown
        if generated_text:
            self.add_markdown(generated_text)
            self.logger.info("Added AI-generated markdown")
        # Log if no text was generated
        else:
            self.logger.error("No text generated from OpenAI API.")


    def add_paragraph_from_df(self, 
        df: pd.DataFrame,
        prompt: str,
        max_rows: Optional[int] = 20) -> None:
        """
        Add a pandas DataFrame as a table with AI-generated explanatory text.

        Args:
            df: Pandas DataFrame to be included as a table
            prompt: Instruction for OpenAI to generate text about the data
            max_rows: Maximum number of rows to include (default: 20)
        """

        # First add the DataFrame as a table (using the existing method)
        self.add_dataframe(df, max_rows)

        # Convert DataFrame to a string format
        # Check if to_markdown is available (requires tabulate)
        try:
            table_str = df.to_markdown(index=False)
        except AttributeError:
            # Fallback to string representation if to_markdown isn't available
            table_str = df.to_string(index=False)

        # Call OpenAI API to generate text about the table
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{prompt}\n\nHere is the data:\n\n{table_str}"}
            ],
            max_tokens=500
        )

        # Extract the generated text
        generated_text = response.choices[0].message.content

        # Add a subheader for the AI analysis
        analysis_header = Paragraph("Analysis:", ParagraphStyle(
            'AnalysisHeader',
            parent=self.styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.darkblue
        ))
        self.elements.append(analysis_header)

        # Add the generated text as markdown
        self.add_markdown(generated_text)
        self.logger.info("Added AI-generated markdown from DataFrame")

    
    def _process_inline_formatting(self, text: str) -> str:
        """Process inline markdown formatting to ReportLab XML-like tags.
        
        Args:
            text: Text with markdown formatting

        Returns:
            str: Text with ReportLab XML-like tags
        """
        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)

        # Italic
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)

        # Code
        text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)

        # Links (simple markdown [text](url))
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<link href="\2">\1</link>', text)

        # Return processed text
        return text


    def _markdown_to_reportlab_elements(self, markdown_text: str) -> List:
        """Convert markdown text to a list of ReportLab flowables.

        Args:
            markdown_text: Text in markdown format

        Returns:
            List: List of ReportLab flowables (Paragraphs, Images, etc.)
        """
        # Initialize list to hold ReportLab elements
        elements = []
        
        # Split markdown text into lines
        lines = markdown_text.split('\n')

        # Initialize tracker variables
        i = 0
        in_code_block = False
        code_block_content = []

        # Process each line
        while i < len(lines):
            line = lines[i].rstrip()

            # Handle code blocks
            if line.startswith('```'):
                # Toggle code block status
                in_code_block = not in_code_block

                # End of code block
                if not in_code_block:
                    code_text = '\n'.join(code_block_content)
                    elements.append(Paragraph(code_text, self.styles['Code']))
                    code_block_content = []

                # Start of code block
                else:
                    code_block_content = []
                i += 1
                continue

            # Inside code block
            if in_code_block:
                code_block_content.append(line)
                i += 1
                continue

            # Headers
            if line.startswith('# '):
                elements.append(Paragraph(self._process_inline_formatting(line[2:].strip()), self.styles['Heading1']))
            elif line.startswith('## '):
                elements.append(Paragraph(self._process_inline_formatting(line[3:].strip()), self.styles['Heading2']))
            elif line.startswith('### '):
                elements.append(Paragraph(self._process_inline_formatting(line[4:].strip()), self.styles['Heading3']))

            # Lists
            elif line.startswith('- ') or line.startswith('* '):
                items = []
                while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                    item_text = self._process_inline_formatting(lines[i][2:].strip())
                    items.append(ListItem(Paragraph(item_text, self.styles['Normal'])))
                    i += 1
                elements.append(ListFlowable(items, bulletType='bullet'))
                continue

            # Empty line
            elif line.strip() == '':
                elements.append(Spacer(1, 0.15*inch))

            # Paragraph
            else:
                elements.append(Paragraph(self._process_inline_formatting(line), self.styles['Normal']))

            # Increment line counter
            i += 1

        # Return the list of ReportLab elements
        return elements
