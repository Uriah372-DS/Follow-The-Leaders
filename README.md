<h1 align="center" style="text-align:center; font-weight:bold; font-size:2.5em">Follow The Leaders</h1>

<p align='center' style="text-align:center;font-size:1em;">
    <a>Uriah Asulin</a>&nbsp;,&nbsp;
    <a>Guy Shoef</a>&nbsp;,&nbsp;
    <a>Omer Yom Tov</a>&nbsp;,&nbsp;
    <br/> 
    Technion – Israel Institute of Technology<br/> 
    
</p>

# Outline
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
   - [Order of Notebooks](#order-of-notebooks)
   - [Additional Directories](#additional-directories)
   - [Getting Started in Databricks](#getting-started-in-databricks)
3. [Project Phases](#project-phases)
   - [1. Data Collection](#1.-data-collection)
   - [2. Data Analysis](#2.-data-analysis)
   - [3. Recommendation Engine](#3.-recommendation-engine)
4. [Running the Notebooks (Contains Important Notice)](#running-the-notebooks)
5. [Dependencies](#dependencies)

---

# Project Overview

"Follow The Leaders" is a pioneering project designed to empower emerging professionals on LinkedIn by providing them with actionable insights derived from the careers of established industry leaders. In today's fast-paced and ever-evolving job market, understanding the paths successful individuals have taken can significantly enhance one's career trajectory.

At its core, this project utilizes advanced data collection techniques to curate profiles of professionals who have made significant strides in their respective fields. Through meticulous analysis, our system identifies common patterns and key milestones in these leaders’ careers, offering users a roadmap tailored to their specific professional aspirations.

The project is structured into several critical phases:

- **Data Collection:** We gather detailed profiles from LinkedIn, focusing on individuals who have demonstrated exemplary career progress and leadership.
- **Data Analysis:** Using unsupervised learning techniques, we analyze these profiles to extract prevalent themes and success patterns.
- **Recommendation Engine:** Leveraging the power of a large language model, we translate these insights into personalized, actionable career advice.

"Follow The Leaders" is more than just a tool—it's a companion for your professional journey. It aims to demystify the steps towards success by providing clear, data-driven pathways and recommendations that help you emulate the career movements of the most successful individuals in your field.

Whether you are just starting out or looking to pivot into a more ambitious role, "Follow The Leaders" offers a unique blend of guidance and inspiration to help you advance your career confidently and effectively.

---

# Project Structure

This project is organized into a series of Databricks notebooks, each serving a specific function within the pipeline from data collection to the visualization of the results. Here is how you can navigate the notebooks:

## Order of Notebooks:

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

## Additional Directories:

- **Utils**
   - `constants.py`: IMPORTANT! Contains constants used across multiple notebooks, such as paths in which to save the data and through which to access project diectories.

- **Local Data**
   - Directory storing raw and cleaned scraped leader profiles for processing.

## Getting Started in Databricks

To explore the notebooks effectively:
1. Clone the repository into your Databricks environment.
2. Navigate through the notebooks in the order listed above to understand the flow of data and operations.
3. Adjust configurations and paths as necessary to match your Databricks setup and data storage (BEFORE running any of the notebooks).

---

# Project Phases
## 1. Data Collection

### Overview

The initial phase of the "Follow The Leaders" project involves a strategic data collection process essential for laying the groundwork for our analysis. We rely on a robust dataset of LinkedIn profiles, provided by BrightData, which is a cornerstone of our project. This dataset was made available to us through our course, ensuring that we had a high-quality and relevant foundation to build upon.

### Rationale

Data collection is pivotal for the success of our project, as the insights and recommendations generated are only as good as the data they are based on. The dataset provided by BrightData includes comprehensive profiles of industry leaders across various sectors. These profiles encompass a range of data points, such as professional experiences, educational backgrounds, and skills, making them ideal for our analytical purposes.

### Process

The data was originally scraped by BrightData, adhering to all ethical and regulatory standards, which ensures that our project rests on a legitimate and trustworthy data foundation. Here’s a brief overview of how we handled the data collection:

1. **Retrieval**: We accessed the dataset through our course resources, which included a pre-arranged agreement with BrightData. This arrangement ensured that the data was not only relevant but also legally and ethically gathered.

2. **Scraping and Cleaning**: Although the initial scraping was performed by BrightData, we conducted additional scraping to update and enrich the profiles with the latest available data. This step was crucial to maintain the currency and applicability of the information. The cleaning process involved removing any inconsistencies and formatting errors, preparing the data for the subsequent analysis phase.

### Importance

This meticulous approach to data collection ensures that "Follow The Leaders" is built on a solid foundation of accurate, up-to-date, and relevant data. By starting with a clear and comprehensive view of the current leaders in various industries, we set the stage for meaningful insights and effective guidance for users aiming to elevate their professional trajectories.

---

## 2. Data Analysis

### Overview

The data analysis phase of "Follow The Leaders" is where the raw data transforms into actionable insights. By applying unsupervised learning techniques, we extract meaningful patterns and trends from the LinkedIn profiles of established leaders. This stage is critical for understanding the underlying factors that contribute to professional success.

### Techniques and Methodologies

#### Textual Data Preparation

The analysis begins with a thorough preparation of the textual data extracted from LinkedIn profiles. This involves tokenizing text, converting to lowercase, and removing punctuation and stopwords to clean and standardize the data. This preprocessing step is crucial for the effectiveness of the subsequent retrieval and topic modeling.

#### Feature Indexing with TF-IDF

We utilize the Term Frequency-Inverse Document Frequency (TF-IDF) method to index the profiles, transforming the textual attributes into a vector space model. This technique highlights the importance of specific terms relative to their frequency across all documents, enabling us to focus on significant words that could indicate leadership traits and successes.

#### Information Retrieval with BM25

For retrieving relevant profiles, we implement the Okapi BM25 scoring function, a widely-used information retrieval technique that ranks profiles based on their relevance to the user’s inputted criteria. This model is particularly adept at handling the nuances of human language in large datasets, making it an ideal choice for our search engine component.

#### Unsupervised Topic Modeling

To distill the essence of the leaders’ profiles, we employ Latent Dirichlet Allocation (LDA), a powerful topic modeling technique. LDA helps us identify common themes and subjects across the corpus, revealing the key topics that are prevalent among industry leaders. This insight allows us to understand what makes these profiles stand out and how they are interconnected.

### Integration of Analytical Results

The culmination of these analytical methods provides a robust foundation for generating personalized career guidance. By understanding the common pathways and notable attributes of successful professionals, our system can recommend specific actions and milestones that align with the user’s career aspirations.

### Significance

This analytical approach not only ensures that our recommendations are grounded in empirical data but also enhances the personalization of the advice provided. Users receive guidance that is not only based on general industry trends but also tailored to mirror the proven paths of successful individuals in their fields of interest.

---

## 3. Recommendation Engine

### Overview

The Recommendation Engine is a critical component of the "Follow The Leaders" project, designed to translate the insights derived from the data analysis phase into personalized, actionable recommendations for our users. This engine not only interprets the data but also provides a clear, structured path for career advancement tailored to individual profiles and aspirations.

### Design and Functionality

#### Integration with Large Language Models (LLM)

At the heart of our Recommendation Engine is the integration of a sophisticated Large Language Model (LLM). This model leverages the processed and analyzed data to generate comprehensive and understandable advice. By utilizing the Gemini Pro model via LangChain, we ensure that our recommendations are not only relevant but also engaging and easy to comprehend.

#### Chain of Thoughts (CoT) Approach

To enhance the LLM’s effectiveness, we employ a "Chain of Thoughts" (CoT) prompting strategy. This involves first providing the LLM with a summarization task based on the topics identified during the topic modeling stage. The summarized context serves as a primer, preparing the LLM to tackle the more complex task of generating a structured career path.

#### Career Path Generation

Once primed, the LLM proceeds to the final and most critical task: building a detailed career path. This path includes specific steps, milestones, and recommendations, which are directly influenced by the themes and patterns identified among industry leaders. 

#### Career Path Visualization

The career path is visualized in a user-friendly format, making it easy for users to follow and understand. We made a simple web UI using Dash & Plotly, that can be viewed in the `Flowchart_colab` notebook in a google-colab environment. It showcases the idea we had in mind then presenting the career path to the user. Our prime motivation for using these tools in our proof-of-concept was that the final result should be interactive so as to not overload the user and to help him focus on one step at a time. Another motivation was showing that our prompt engineering gave a structured LLM response that can be parsed and then visualized inside predefined web components, which requires the LLM to be predictable and concise.

### Personalization

Our engine personalizes recommendations by considering the user's specific inputs regarding their career goals, desired positions, and preferred industries. This personalization ensures that the advice is not only based on generic data but is finely tuned to meet the individual's unique career objectives.

### Impact and Utility

The Recommendation Engine is designed to be a transformative tool for career development. By providing tailored advice based on proven pathways of successful professionals, it empowers users to make informed decisions that can significantly enhance their career prospects. Whether users are just starting out or looking to pivot to more ambitious roles, the engine offers them a clear and practical guide towards achieving their goals.

---

# Running the Notebooks

*IMPORTANT!*
Before running any of the notebooks you must edit the values of the constants in `Utils` -> `constants.py`. More detailed instructions inside the `constants.py` file.

Each notebook is designed to be semi-self-contained but follows the logical progression described above. Ensure you run the notebooks in the specified order to maintain data integrity and flow throughout the project.
This is exluding the Flowchat_colab notebook, which can be run entirely independently of the rest of the notebooks in a google-colab environment.

# Dependencies

Ensure you have Python installed, along with the library:
- `pip install sparkml-base-classes  # You can simply add it to the local / global init script of you cluster, or manually add it in every notebook that you wish to run`

Optional:

For running the notebook `Testing Open-Source Tools` inside the search_engine directory, add these lines into your init script:
```# Directory where libraries will be stored
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
```

---
