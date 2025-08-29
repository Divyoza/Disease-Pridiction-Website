from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Create chatbot
chatbot = ChatBot("HealthBot")

# Train data (you can expand this later)
diabetes_data = [
    "What is diabetes?",
    "Diabetes is a condition where the body cannot properly process blood sugar.",
    "What are symptoms of diabetes?",
    "Common symptoms include frequent urination, thirst, and fatigue.",
    "How can I control diabetes?",
    "You can control diabetes with diet, exercise, medication, and regular monitoring.",
    "What food should I avoid in diabetes?",
    "Avoid sugary drinks, fried foods, and refined carbs."
]

heart_disease_data = [
    "What is heart disease?",
    "Heart disease refers to conditions that affect the heart’s structure and function.",
    "What are symptoms of heart disease?",
    "Chest pain, shortness of breath, and fatigue are common symptoms.",
    "How can I reduce the risk of heart disease?",
    "Exercise regularly, eat healthy, and avoid smoking.",
    "What food is good for heart patients?",
    "Whole grains, fruits, vegetables, and lean proteins are good."
]

cancer_data = [
    "What is cancer?",
    "Cancer is a disease where abnormal cells grow uncontrollably in the body.",
    "What are symptoms of cancer?",
    "Unexplained weight loss, fatigue, and lumps are common symptoms.",
    "How can cancer be treated?",
    "Cancer can be treated with surgery, chemotherapy, radiation, or immunotherapy."
]

# Train chatbot
trainer = ListTrainer(chatbot)
trainer.train(diabetes_data)
trainer.train(heart_disease_data)
trainer.train(cancer_data)

print("Chatbot training complete!")
