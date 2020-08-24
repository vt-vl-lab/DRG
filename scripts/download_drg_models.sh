#!/bin/bash

# Download VCOCO models
echo "Downloading VCOCO models"
mkdir -p output/VCOCO
python scripts/Download_data.py 1sFZqbBUrvE9jtuihto-G7Z0VZAwT5U0g output/VCOCO/model_app.pth
python scripts/Download_data.py 1lv2vHUEW0IADUikR2zBABnyJtH2QnMxJ output/VCOCO/model_sp_human.pth
python scripts/Download_data.py 1cBd6Y82CBnIVruLjGTYBM8efWLBu8OAd output/VCOCO/model_sp_object.pth

# Download HICO-DET models
echo "Downloading HICO-DET models"
mkdir -p output/HICO
python scripts/Download_data.py 1Z1Fall2x0yVoN2TBlTOeTzGML_0pZCqZ output/HICO/model_app.pth
python scripts/Download_data.py 1jGfeD9RXCXOFdjyD-reyAGtspVfc4QgL output/HICO/model_sp_human.pth
python scripts/Download_data.py 193UnCSlU_JwD9RTBnX8seDg09X4iWNqq output/HICO/model_sp_object.pth