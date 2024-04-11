# Follow-The-Leaders

**Follow-The-Leaders** is a LinkedIn feature designed to guide users through a timeline of events, helping them achieve their professional goals efficiently.
This project aims to assist emerging professionals on LinkedIn by analyzing the profiles of established leaders in their fields, which are selected based on user-provided queries. By examining the career trajectories and qualifications of these leaders, the project provides actionable insights and recommendations to users seeking to advance their own careers.

## Project Structure

This project is organized into a series of Databricks notebooks, each serving a specific function within the pipeline from data collection to the visualization of the results. Here is how you can navigate the notebooks:

### Order of Notebooks:

1. **Data Collection**
   - `collecting_target_users.py`: Script for collecting LinkedIn profile URLs of potential leaders.

2. **Data Preprocessing**
   - `Analyzing Scraped Leaders.py`: Notebook for analyzing and preprocessing the scraped data.

3. **Search Engine**
   - `BM25 Search Engine.py`: Implements the BM25 search engine for retrieving relevant profiles.

4. **Model Development**
   - `Unsupervised Topic Modeling.py`: Notebook for performing topic modeling on the leader profiles using the LSA algorithm.
   - `LLM integration.py`: Integrates a large language model for generating structured career paths based on analyzed data.

5. **Visualization**
   - `Flowchart_colab`: Optional interactive visualization (run in Google Colab) that displays end-user results.

6. **Evaluation**
   - `Search Engine Evaluation using Clustering Methods.py`: Evaluates the search engine performance using clustering methods.

### Additional Directories:

- **Utils**
   - `constants.py`: IMPORTANT! Contains constants used across multiple notebooks, such as paths in which to save the data and though which to access project diectories.

- **Local Data**
   - Directory storing raw and cleaned scraped leader profiles for processing.

### Getting Started in Databricks

To explore the notebooks effectively:
1. Clone the repository into your Databricks environment.
2. Navigate through the notebooks in the order listed above to understand the flow of data and operations.
3. Adjust configurations and paths as necessary to match your Databricks setup and data storage (BEFORE running any of the notebooks).

## Running the Notebooks

*IMPORTANT!*
Before running any of the notebooks you must edit the following values of the constants in `Utils` -> `constants.py`

Each notebook is designed to be semi-self-contained but follows the logical progression described above. Ensure you run the notebooks in the specified order to maintain data integrity and flow throughout the project.
This is exluding the Flowchat_colab notebook, which can be run entirely independently of the rest of the notebooks in a google-colab environment.

### Dependencies

Ensure you have Python installed, along with the library:
- pip install sparkml-base-classes  # You can simply add it to the local / global init script of you cluster, or manually add it in every notebook that you wish to run

Optional:

For running the notebook `Testing Open-Source Tools` inside the search_engine directory, add these lines into your init script:
`# Directory where libraries will be stored
lib_dir="/databricks/jars"

# Create the directory if it doesn't exist
mkdir -p "$lib_dir"

# Download the library JAR - replace the URL with the actual JAR location
curl -o $lib_dir/spark-lucenerdd.jar "https://repo1.maven.org/maven2/org/zouzias/spark-lucenerdd_2.12/0.4.0/spark-lucenerdd_2.12-0.4.0.jar"

# Download Lucene core and other necessary libraries
curl -L -o $lib_dir/lucene-core-8.11.2.jar "https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/8.11.2/lucene-core-8.11.2.jar"
curl -L -o $lib_dir/lucene-facet-8.11.2.jar "https://repo1.maven.org/maven2/org/apache/lucene/lucene-facet/8.11.2/lucene-facet-8.11.2.jar"
curl -L -o $lib_dir/lucene-analyzers-common-8.11.2.jar "https://repo1.maven.org/maven2/org/apache/lucene/lucene-analyzers-common/8.11.2/lucene-analyzers-common-8.11.2.jar"
curl -L -o $lib_dir/lucene-queryparser-8.11.2.jar "https://repo1.maven.org/maven2/org/apache/lucene/lucene-queryparser/8.11.2/lucene-queryparser-8.11.2.jar"
curl -L -o $lib_dir/lucene-expressions-8.11.2.jar "https://repo1.maven.org/maven2/org/apache/lucene/lucene-expressions/8.11.2/lucene-expressions-8.11.2.jar"
curl -L -o $lib_dir/lucene-spatial-extras-8.11.2.jar "https://repo1.maven.org/maven2/org/apache/lucene/lucene-spatial-extras/8.11.2/lucene-spatial-extras-8.11.2.jar"

# Algebird
curl -L -o $lib_dir/algebird-core_2.12-0.13.10.jar "https://repo1.maven.org/maven2/com/twitter/algebird-core_2.12/0.13.10/algebird-core_2.12-0.13.10.jar"

# Spatial4j
curl -L -o $lib_dir/spatial4j-0.8.jar "https://repo1.maven.org/maven2/org/locationtech/spatial4j/spatial4j/0.8/spatial4j-0.8.jar"

## JTS Core
curl -L -o $lib_dir/jts-core-1.19.0.jar "https://repo1.maven.org/maven2/org/locationtech/jts/jts-core/1.19.0/jts-core-1.19.0.jar"

# Joda-Time
curl -L -o $lib_dir/joda-time-2.12.5.jar "https://repo1.maven.org/maven2/joda-time/joda-time/2.12.5/joda-time-2.12.5.jar"

# Joda-Convert
curl -L -o $lib_dir/joda-convert-2.2.3.jar "https://repo1.maven.org/maven2/org/joda/joda-convert/2.2.3/joda-convert-2.2.3.jar"

# Typesafe Config
curl -L -o $lib_dir/config-1.3.4.jar "https://repo1.maven.org/maven2/com/typesafe/config/1.3.4/config-1.3.4.jar"

# Scalactic
curl -L -o $lib_dir/scalactic_2.12-3.2.17.jar "https://repo1.maven.org/maven2/org/scalactic/scalactic_2.12/3.2.17/scalactic_2.12-3.2.17.jar"

# Scalatest
curl -L -o $lib_dir/scalatest_2.12-3.2.17.jar "https://repo1.maven.org/maven2/org/scalatest/scalatest_2.12/3.2.17/scalatest_2.12-3.2.17.jar"
`
