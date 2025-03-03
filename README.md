NutriBuddy – A Diet Plan Recommendation System
Overview
NutriBuddy is an AI-powered chatbot designed to predict chronic diseases based on symptoms and provide personalized diet recommendations. The system leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to enhance accuracy and provide relevant dietary suggestions. The chatbot can predict diseases like diabetes, heart disease, and thyroid disorders based on symptoms and suggest suitable diet plans accordingly.

Key Features
✅ Disease Prediction
Users can enter symptoms, and the chatbot will predict possible diseases.
Utilizes LLM-based reasoning and vector embeddings to enhance accuracy.
🍽 Diet Plan Recommendations
Provides personalized diet plans based on the detected disease.
Suggests nutrient-rich food options and lists items to avoid.
🗣 Conversational AI
Interactive chatbot interface for real-time user engagement.
Users can ask follow-up questions about their diet plan.
📊 Data-Driven Insights
Uses a vector database (Chroma DB) to store and retrieve medical and dietary research.
Leverages PostgreSQL to fetch detailed nutritional data.
🔍 AI-Powered Query Processing
Hugging Face’s all-mpnet-base-v2 embedding model converts textual data into vector format for improved search accuracy.
Integrates LangChain to create structured queries for the LLM.
Tech Stack
Component	Technology Used
Backend	Flask (Python-based web framework)
AI Model	Llama 3.1 8B (Meta)
Data Storage	PostgreSQL (Food dataset), Chroma DB (Vector storage)
Embedding Model	Hugging Face all-mpnet-base-v2
AI Framework	LangChain, Ollama
Project Architecture
User Input Handling:

The user enters either symptoms or selects a known disease.
Disease Prediction (if symptoms are provided):

The chatbot searches the vector database for relevant disease information.
The LLM processes the input and predicts the disease.
Diet Plan Generation:

Based on the identified disease, the system retrieves relevant dietary research from Chroma DB.
The LLM structures a diet plan, including:
Nutrient recommendations
Suggested food items
Foods to avoid
Follow-up Questions & Interaction:

Users can ask further questions, and responses are context-aware using stored conversation history.
Installation Guide
Prerequisites
Ensure you have the following installed before running the project:

Python 3.8+
PostgreSQL (For storing nutritional data)
Ollama (For running Llama 3.1 model locally)
Step 1: Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/nutribuddy.git
cd nutribuddy
Step 2: Set Up the Environment
Install the required dependencies:

bash
Copy
Edit
pip install flask langchain psycopg2 chromadb transformers
Step 3: Database Setup
PostgreSQL: Create a database and import the food nutrition dataset.
Chroma DB: Load the research papers and store vector embeddings for disease prediction and diet recommendations.
Step 4: Run the Application
bash
Copy
Edit
python app.py
The Flask app should now be running on http://127.0.0.1:5000/.

Usage Guide
1️⃣ Disease Prediction Mode
If users are unsure about their condition, they can input symptoms.
The system will analyze the symptoms and predict the most likely disease.
2️⃣ Direct Diet Recommendation Mode
If users already know their disease, they can select it directly.
The chatbot will retrieve the best dietary recommendations based on medical research.
3️⃣ Interactive Conversations
Users can ask follow-up questions to clarify or refine their diet plan.
The chatbot maintains context awareness for better recommendations.
Sample API Endpoints
1️⃣ Predict Disease from Symptoms
http
Copy
Edit
POST /predict-disease
Request Body
json
Copy
Edit
{
  "symptoms": "fatigue, dry mouth, itching"
}
Response
json
Copy
Edit
{
  "predicted_disease": "Diabetes"
}
2️⃣ Get Diet Plan for a Disease
http
Copy
Edit
POST /get-diet-plan
Request Body
json
Copy
Edit
{
  "disease": "Diabetes"
}
Response
json
Copy
Edit
{
  "diet_plan_summary": "A balanced diet including low-carb vegetables, lean proteins, and whole grains...",
  "nutrient_recommendations": ["Fiber", "Protein", "Omega-3"],
  "foods_to_avoid": ["Sugary drinks", "White bread", "Processed snacks"]
}
Results & Evaluation
Accuracy: The chatbot achieves 90% accuracy in predicting diseases and suggesting suitable diets.
Consistency: The system generates consistent responses for similar inputs.
Non-Hallucinating Responses: The chatbot only provides diet plans from the contextual database to avoid misinformation.
Example Use Case
Input	Response
Symptoms: "Fatigue, dry mouth, itching"	Predicted Disease: Diabetes
Disease: Diabetes	Diet Plan: High-fiber foods, lean protein, no processed sugars
Future Enhancements
🚀 Additional Features Coming Soon!

Integration with professional dietitian datasets for more refined food suggestions.
Detailed calorie-based meal plans using USDA FoodData Central.
Voice-based chatbot functionality for hands-free usage.
Multi-disease expansion to include more chronic conditions beyond diabetes, heart disease, and thyroid disorders.
Contributors
This project was developed by a team of graduate students from the University of North Texas:

Chittal Karuppiah
Sahika Reddy Palvai
Sri Harsha Sepuri
Sai Yashwanth Ranga 🚀
For inquiries, reach out to SaiYashwanthRanga@my.unt.edu.

License
This project is licensed under the MIT License – feel free to use and contribute!

