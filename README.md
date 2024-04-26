# Knowledge Graph

In this Master thesis Project, a Knowledge Graph about stakeholder of Bavarian Ministries is constructed. Furthermore this work predicts future relations of stakeholders. Thus, the following steps are taken: 

1. Building of the webscraper (webscraper_sql): 
  - Scraping of pages, their connected links and the underlying pages of all 12 ministries
  - textscraping of all websites including PDF files

2. Data cleaning and named-entity-recognition with FLAIR NLP (ner_sql)
  - Cleaning of abbreviations and similar entities
  - Named entity recognition with FLAIR NLP to identify personal names and organisations
  - Counter to check the number of occurrence of names or organisations

3. Construction, visualisation and analysis of the stakeholder network (neo4jgraph)
  - Loading the dataset into neo4j graph data using a Neo4J docker image
  - Visualisation of the graph with Neo4J Browser and Gephi
  - Network Analysis of clustering, degree and centralities

4. Link prediction using Machine Learning techniques
  - Time-series forecasting of features
  - Topological link prediction method
  - Validation of both models using k-fold time-aware cross-validation
  - Combining both ML approaches to a hybrif approach 
  - Prediction of future links

# How to use the code
To replicate the code and its results, please follow these steps: 
  - Use Python 3.11, some packages are not yet working in 3.12
  - Download all all packages from requirements.txt
  - To use the webscraper: 
    - Configure all data you want to access in config.yml (especially a list of base addresses)
    - Check if you want to include PDF scraping and if you want to stop at a certain website click depth
    - Comment out code parts if you do not need these
    - run webscraper_sql/main.py
  - To use the cleaning and named-entity-recognition:
    - Prerequisits: You need text data from the webscraper in sqllite format (csv does also work, but changes on the code are needed) and the org_count_list.csv. The database is called scraped_data.db. 
    - Check that your data is structured like the sqllite 3 database of the webscraper
    - Download the model of FLAIR ner-german-large: https://huggingface.co/flair/ner-german-large
    - Run the ner_sql/main.py
  - To import the data into the neo4j database and conduct statistical analysis:
    - Have the final data of named-entity-recognition as sqllite 3 database (Here: ner-data-sim.db) and the scraped_data.db
    - Pull the neo4j image from Docker
    - Configure the docker dompose file accordingly and run docker compose up
    - Now run the neo4jgraph/main.py file to load the sqldatabase to neo4j graph database
    - Run the last Neo4J queries in queries_sql.py directly in Neo4jBrowser to change the visualisation of the graph
    - Run queries in stat_analysis directly in Neo4jBrowser to get the results
    - For final visualisation, import the neo4j data in Gephi and configure the view
  - To make the link predictions: 
    - Firstly run time_series_train.py to forecast the features. Configure which years should be calculated and the split you want to run in the config file. 
    - Validate the time-series forecasting with time_series_val.py and predict the features of 2025. 
    - Run link_pred_train_classify.py and configure the split you want to construct in the config file.
    - Validate the link prediction classifier on the validation split of the best performing model with link_pred_val.py.
    - Validate and run the final hybrid link prediction using the script final_link_prediction.py



## Master thesis project

The knowledge graph project is a master thesis project and the aim is to make the data that is collected publicly accessible in the open bydata portal. 


## Steps taken and Roadmap:

- [X] Build a webscraper from scratch (November)
- [X] Run webscraper in AWS Instance and receive a final dataset (December)
- [X] Clean up data before processing it further (December)
- [X] Perform successful named entity recognition (January)
- [X] Import final data in neo4j with stakeholders and relations (February 15th)
- [X] Make network analysis in neo4j (February)
- [X] Visualize the network and the analysis accordingly (March)
- [X] Build Machine Learning Model for link prediction (March)
- [X] Build a second time-series forecasting model (April)
- [X] Make final predictions with hybrid approach (April)
- [X] Final release end of April 2024
- [ ] Host on open data plattform (hopefully in May)

## Bei Fragen und Anregungen: 
Gerne bei Lena melden :)


