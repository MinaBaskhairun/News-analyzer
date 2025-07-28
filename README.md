# 📰 News Article Analyzer

A simple Python project that fetches news articles using NewsAPI, analyzes them with spaCy, and visualizes named entities like people, organizations, and places.

## Features
- Fetch articles by keyword (e.g., "AI")
- Extract and count named entities
- Display top entities in a bar chart
- Generate a word cloud

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
