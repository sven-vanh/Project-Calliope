# MAIN
# This script demonstrates the functionality of the PDFGenerator class.
# It creates a PDF report with headers, paragraphs, figures, tables, AI analysis, and page breaks.
# Created By: Sven van Helten
# Date: 17.04.2025
# Version: 1.0
# License: MIT

# Import dependencies
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from src.pdf_generator import PDFGenerator
import logging

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables with fallbacks
API_KEY = os.getenv("LLM_API_KEY")
API_BASE_URL = os.getenv("LLM_API_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "gemma3:4b-it-q8_0")
SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are a data analyst.")

def main():
    """
    Demo for PDFGenerator: tests headers, paragraphs, figures, tables, AI analysis, and page breaks.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    print("==== PDFGenerator Demo ====")

    # Initialize generator with environment variables
    pdf = PDFGenerator(
        "demo_report.pdf",
        api_key=API_KEY,
        api_base_url=API_BASE_URL,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT
    )

    # Add header and intro paragraph
    pdf.add_header("Demo Report")
    pdf.add_paragraph("This report demonstrates all PDFGenerator functionality with demo data.")

    # Create and add a simple matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [4, 1, 3, 5], marker='o')
    ax.set_title("Sample Line Plot")
    pdf.add_figure(fig)

    # Create and add a pandas DataFrame
    data = {
        "Name": ["Alice", "Bob", "Charlie", "Diana"],
        "Score": [85, 92, 78, 88]
    }
    df = pd.DataFrame(data)
    pdf.add_dataframe(df)

    # Page break before AI analyses
    pdf.add_page_break()

    # AI-generated analysis for figure
    pdf.add_paragraph_from_figure(fig, "Provide a brief analysis of the following line plot.")

    # AI-generated analysis for DataFrame
    pdf.add_paragraph_from_df(df, "Summarize the performance scores in the following data table.")

    # Save the PDF
    pdf.save()
    print("Demo PDF generated: demo_report.pdf")
    print("==== End of Demo ====")


if __name__ == "__main__":
    main()
