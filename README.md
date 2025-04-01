# Real Estate Investment Recommendation Bot

This project implements a multi-agent system powered by LangChain to deliver comprehensive real estate investment recommendations. By integrating structured data, current market insights, demographic trends, and custom code generation, the system leverages six specialized agents to create a 20-page investment report packed with data visualizations, listings, reviews, and analytical insights.

---

## **📌 Project Overview**

The system orchestrates six distinct agents to cover every aspect of real estate research:

1. **Investment Analytics Agent:**  
   - Connects to Snowflake to query structured real estate data.
   - Generates graphs and visualizations to showcase historical trends and investment metrics.

2. **Investment Listing Agent:**  
   - Retrieves current housing listings, blogs, and reviews from Snowflake.
   - Provides real-time market snapshots to support investment decisions.

3. **Market Sentiment Agent:**  
   - Leverages Pinecone for semantic search and vector-based data retrieval.
   - Enhances the recommendation process through efficient similarity search and contextual data filtering.

4. **Code Generator Agent:**  
   - Automates the creation of code snippets to facilitate data processing and visualization.
   - Supports dynamic report generation and on-the-fly customization.

5. **Demographic Agent:**  
   - Aggregates and analyzes demographic data relevant to real estate trends.
   - Offers insights into population trends, income levels, and neighborhood characteristics that influence investment potential.

6. **Final Report Agent:**  
   - Combines outputs from all the other agents.
   - Generates a cohesive 20-page comprehensive report featuring visualizations, listings, analytical insights, and recommendations.

---

## **📌 Project Resources**
- **User Interface:**  
  - Interactive dashboards for visualizing graphs, listings, and demographic trends.
- **Backend Services:**  
  - RESTful APIs to orchestrate agent communication and data aggregation.
- **Documentation & Demo:**  
  - Detailed technical documentation and demo videos are available in the project repository.

---

## **📌 Technologies Used**

<p align="center">
  <img src="https://img.shields.io/badge/-LangChain-000000?style=for-the-badge" alt="LangChain">
  <img src="https://img.shields.io/badge/-Snowflake-007FFF?style=for-the-badge" alt="Snowflake">
  <img src="https://img.shields.io/badge/-Pinecone-734BD4?style=for-the-badge" alt="Pinecone">
  <img src="https://img.shields.io/badge/-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

---


## **📌 Project Flow**

### **Step 1: Data Collection & Requirements**
- **Structured Data:**  
  - Real estate metrics, housing listings and Demographic insights stored in Snowflake.
- **Unstructured Data:**  
  - blogs, and reviews sourced from various data feeds.
- **Vector Data:**  
  - Semantic search enabled by Pinecone for contextual relevance.

### **Step 2: Agent Setup & Responsibilities**

#### **A. Snowflake Agents**
- **Graph Agent:**  
  - Extracts structured data and renders data visualizations.
- **Listing Agent:**  
  - Retrieves current housing listings and market reviews.

#### **B. Pinecone Agent**
- **Functionality:**  
  - Provides fast, vector-based retrieval for similarity search.
  - Supports filtering and contextual search for enhanced data relevance.

#### **C. Code Generator Agent**
- **Functionality:**  
  - Automatically produces code snippets to streamline data analysis and visualization generation.
  
#### **D. Demographic Agent**
- **Functionality:**  
  - Analyzes population and economic indicators to provide demographic context.
  
#### **E. Final Report Agent**
- **Functionality:**  
  - Integrates outputs from all agents.
  - Generates a comprehensive 20-page investment report with consolidated insights, visuals, and recommendations.

### **Step 3: Report Generation & UI Integration**
- **Report Generation:**  
  - Aggregates data from all agents into a detailed, multi-page report.
- **User Interface:**  
  - Utilizes Streamlit and FastAPI for an interactive experience.
  - Allows users to trigger individual agents or the combined final report generation.

### **Step 4: Deployment**
- **Deployment Environment:**  
  - Dockerized setup for seamless integration and deployment.
- **Scalability:**  
  - Designed to scale and adapt to additional data sources and analytic requirements.

---

## **📌 Contributions**

| **Member**    | **Contribution**                                                                                                                                                         |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Ketki**  | Developed the Snowflake Graph and Listing Agents for data extraction and visualization from real estate datasets.                                                        |
| **Janvi**  | Implemented the Pinecone Agent for semantic search and assisted in integrating the Code Generator Agent for dynamic code snippet generation.                           |
| **Sahil**  | Built the Demographic Agent to analyze and integrate demographic data, and designed the Final Report Agent to aggregate and compile a comprehensive 20-page report.    |

---

## **📌 Attestation**

**WE CERTIFY THAT WE HAVE NOT USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND COMPLY WITH THE POLICIES OUTLINED IN THE STUDENT HANDBOOK.**
