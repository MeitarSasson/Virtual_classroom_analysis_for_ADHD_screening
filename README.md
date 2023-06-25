# Virtual_classroom_analysis_for_ADHD_screening
Our project involves a machine learning model (XGBoost Gradient Boosting).<br>
It has a CouchDB Database storage implementation.<br>
This is a binary classification model, which can be trained to predict if a test subject has ADHD suspicion.<br>
We also have a students registry which are the test subjects.<br>
A student can register if he/she has a JSON file associated with them.<br>
The JSON files, which are stored in CouchDB, represent the test subjects during 2 test sessions - with and without disturbances.<br>
We also have various plots to show the results of the dataframes, the JSON files structure and the results.<br>
We hope you'll have a great time exploring our project!

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Getting Started 

You should be able to download and install CouchDB from https://couchdb.apache.org/#download

### Prerequisites


While installing CouchDB, you need to keep the cookie, username and password so that you can access them later.

### Installation 

commandline
pip install -r requirements.txt

## Usage

commandline
python main.py

## Authors 

- Meitar Sasson - Meitar27797@gmail.com 
- Shimon Elkobi - Shoham1994@gmail.com
