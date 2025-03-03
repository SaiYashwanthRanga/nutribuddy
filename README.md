NutriBuddy – A Diet Plan Recommendation System
Overview
NutriBuddy is an AI-powered chatbot designed to predict chronic diseases based on symptoms and provide personalized diet recommendations. The chatbot leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to offer accurate and flexible dietary guidance.

Features
Disease Prediction: Users can input symptoms to get a probable disease diagnosis.
Diet Plan Recommendations: Personalized diet plans for heart disease, diabetes, and thyroid conditions.
Conversational AI: A chatbot interface for user interaction.
Vector Database: Uses Chroma DB for storing research papers and diet-related information.
Food Data Integration: Retrieves nutritional data from PostgreSQL database.
Tech Stack
Backend: Python, Flask
LLM Model: Llama 3.1 8B (Meta)
Database: PostgreSQL, Chroma DB
Vector Embeddings: Hugging Face’s all-mpnet-base-v2
Frameworks: LangChain, Ollama
Installation
Prerequisites
Python 3.8+
PostgreSQL
Ollama (for running Llama 3.1 locally)
Required Python packages:
bash
Copy
Edit
pip install flask langchain psycopg2 chromadb transformers
Setup
Clone the repository:
bash
Copy
Edit
git clone https://github.com/your-username/nutribuddy.git  
cd nutribuddy  
Set up the database in PostgreSQL and import the food dataset.
Run the Flask application:
bash
Copy
Edit
python app.py  
Usage
Access the chatbot via the web interface.
Enter symptoms or select a disease for diet recommendations.
Follow up with additional queries to refine dietary advice.
Future Enhancements
Voice-based chatbot support.
Integration of real-time health data.
Expansion to cover more diseases and dietary plans.
