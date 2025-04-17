# Project Calliope

A Python library for generating professional PDF reports with AI-enhanced content analysis.

## Overview

Project Calliope provides a simple yet powerful interface for creating PDF reports that combine data visualizations, tables, and AI-generated insights. The core component is the `PDFGenerator` class, which wraps ReportLab functionality with a streamlined API and adds AI capabilities via LLM integration.

## Key Features

- Generate professionally styled PDF reports with consistent formatting
- Include headers, paragraphs, page breaks, and support for basic Markdown
- Add matplotlib figures with automatic styling
- Convert pandas DataFrames to formatted tables
- Generate AI-powered analysis of figures and data tables
- Flexible LLM backend support (OpenAI API or local models via Ollama)

## Components

### PDFGenerator Class

The `PDFGenerator` class (`src/pdf_generator.py`) is the primary component, offering methods to:

- Add headers and text paragraphs
- Insert matplotlib figures with customizable dimensions
- Convert pandas DataFrames to formatted tables
- Add AI-generated analysis of figures and data
- Process Markdown formatting
- Control document layout with page breaks

## Installation

```bash
# Clone the repository
git clone https://github.com/tim-vanh/project-calliope.git
cd project-calliope

# Install dependencies
uv sync
```

## Usage Example

```python
from src.pdf_generator import PDFGenerator
import matplotlib.pyplot as plt
import pandas as pd

# Initialize generator
pdf = PDFGenerator(
    "report.pdf", 
    api_key="your-api-key",
    # Optional parameters for an OpenAI-compatible API
    api_base_url="http://localhost:11434/v1",
    model="gemma3,
    system_prompt="You are a data analyst."
)

# Add content
pdf.add_header("Analysis Report")
pdf.add_paragraph("This report contains data analysis and visualizations.")

# Add a matplotlib figure
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 15, 13, 17])
pdf.add_figure(fig)

# Add a table from pandas DataFrame
data = {"Name": ["Alice", "Bob"], "Score": [95, 87]}
df = pd.DataFrame(data)
pdf.add_dataframe(df)

# Add AI analysis of the figure
pdf.add_paragraph_from_figure(fig, "Analyze this trend data.")

# Add AI analysis of the DataFrame
pdf.add_paragraph_from_df(df, "Summarize this performance data.")

# Save the PDF
pdf.save()
```

## Using with Ollama

Project Calliope can use local LLMs through [Ollama](https://ollama.ai/) or any other OpenAI-compatible API:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Start the Ollama server
3. Pull your preferred vision-enabled language model (e.g., `ollama pull gemma3:4b-it-q8_0`)
4. Configure the PDFGenerator with:
   - `api_key="ollama-demo"` (or any string)
   - `api_base_url="http://localhost:11434/v1"`
   - `model="gemma3:4b-it-q8_0"` (or your chosen model name)

## Dependencies

- reportlab - PDF generation
- matplotlib - Data visualization
- pandas - Data handling
- openai - API client for LLM integration

## License

Project Calliope is released under the [MIT License](LICENSE).
