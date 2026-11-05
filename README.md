# PCA & SVD Virtual Lab

A browser-based interactive virtual lab built with Streamlit.

## Features
- Iris dataset included (no download required)
- Upload a CSV and choose numeric features
- PCA: 2D/3D projection, explained variance, downloadable projection
- SVD: rank-k reconstruction, singular values, reconstruction error, downloadable output
- Theory, procedure, observation and conclusion tabs

## Run locally (Windows)
1. Install Python 3.10+.
2. Extract this ZIP.
3. Open Command Prompt in the extracted folder.
4. Run:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Open the local URL shown by Streamlit (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app**, select the repository, branch, and `app.py`.
4. Click **Deploy**.
5. Share the generated public app URL with students.

## Notes
- For uploaded CSV data, choose numeric columns only. Optional label column is used only for coloring points.
- The app standardizes selected features before PCA/SVD.
- This is an educational virtual lab; results depend on the chosen dataset and features.
