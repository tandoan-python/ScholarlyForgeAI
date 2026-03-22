# ScholarlyForge AI

## Overview

ScholarlyForge AI is an advanced framework generation tool engineered specifically for students and early career researchers. By synthesizing abstract concepts and experimental source code into structured scientific paper outlines, this system automates the creation of LaTeX templates and BibTeX bibliographies. This automation allows researchers to dedicate their cognitive resources to core methodologies and empirical validation.

## System Prerequisites

The application requires a baseline environment of Python 3.10 or Python 3.11. For systems utilizing the Chocolatey package manager, install the necessary multimedia and typesetting dependencies via the following command:

```
choco install ffmpeg miktex
```


## Installation Instructions

Execute the following sequence to isolate dependencies and configure the operational environment.

**1. Virtual Environment Initialization**

Establish a secure, isolated Python environment using the terminal:

```
python -m venv env

```


**2. Environment Activation**

Activate the virtual environment corresponding to your operating system:

For Windows:

```
env\Scripts\activate

```

For macOS and Linux:

```
source env/bin/activate
```


**3. Dependency Installation**

Ensure the `requirements.txt` file is located within the active directory. Execute the following command to install all mandatory libraries:

```
pip install -r requirements.txt
```


To force a comprehensive update of all currently installed packages, execute:

```
pip install -U -r requirements.txt
```


## Application Configuration

The system utilizes the Google Gemini API for natural language reasoning and analytical processing.

Create a file named `.env` in the root directory of the application.

Define the API key variable precisely as follows:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```


## Execution Protocol

Launch the Streamlit web interface by executing the following command in the terminal:

```
streamlit run app.py
```


## License and Attribution

This software is distributed under the MIT License. It remains free and open source to support the academic community. Any utilization, modification, or distribution of this codebase requires explicit attribution to the original creator, Thanh-Tan Doan (tandoan-python github).
