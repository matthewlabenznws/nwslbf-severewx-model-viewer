name: REFS M05 Reflectivity UH

on:
  workflow_dispatch:

jobs:
  run-refs-m03-refl-uh:
    runs-on: ubuntu-latest
    timeout-minutes: 180

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install system packages
        run: |
          sudo apt-get update
          sudo apt-get install -y libeccodes0

      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run REFS M03 Reflectivity UH
        env:
          R2_ACCOUNT_ID: ${{ secrets.R2_ACCOUNT_ID }}
          R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY_ID }}
          R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_ACCESS_KEY }}
          R2_BUCKET: ${{ secrets.R2_BUCKET }}
        run: |
          python refs_m05_refl_uh.py
