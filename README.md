# rowing

## Developer Guide

Install `poetry` to manage dependencies and your virtual environment.

```bash
# Add dependencies
poetry install 

# Use the virtual environment
poetry shell

streamlit run src/Peach_analysis.py
```

To run the app as a Docker image (as it is deployed on Heroku).

```bash
docker build -t rowing:latest .
docker run rowing:latest -p 8501:8501
```
