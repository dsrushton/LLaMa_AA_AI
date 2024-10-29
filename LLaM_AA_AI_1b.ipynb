from collections import deque
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the LLaMA model and tokenizer
model_path = r"C:\Users\dsrus\.llama\checkpoints\Llama3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Load Q/A data
qa_data_path = r"C:\Users\dsrus\Desktop\Workspace\MTLiens\12+12_QA_with_value_groups.json"
with open(qa_data_path, 'r') as f:
    qa_data = json.load(f)

print(f"Loaded {len(qa_data)} Q/A pairs")

# Load Document Corpus
reading_path = r"C:\Users\dsrus\Desktop\Workspace\Google Drive Sync\Shared Docs\Cleaned"
reading_data = []
for filename in os.listdir(reading_path):
    if filename.endswith(".txt"):
        with open(os.path.join(reading_path, filename), "r", encoding="utf-8") as file:
            content = file.read()
            # Split content into smaller chunks (e.g., paragraphs or fixed-size chunks)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            reading_data.extend(chunks)

print(f"Loaded {len(reading_data)} chunks of additional reading material")

# Initialize sentence transformer for embeddings
#embed_model = SentenceTransformer('all-MiniLM-L6-v2')  #####TWO VERSIONS MINI FASTER
embed_model = SentenceTransformer('all-mpnet-base-v2')  #####TWO VERSIONS MPNET SLOWER BUT BETTER I GUESS LOL

# Create embeddings for all Q/A pairs and reading material
def create_embeddings(data):
    embeddings = []
    for item in data:
        if isinstance(item, dict):  # Q&A pair
            text = f"Question: {item['question']} Answer: {item['answer']}"
        else:  # Reading material chunk
            text = item
        embedding = embed_model.encode(text)
        embeddings.append(embedding)
    return np.array(embeddings)

qa_embeddings = create_embeddings(qa_data)
reading_embeddings = create_embeddings(reading_data)

all_embeddings = np.vstack((qa_embeddings, reading_embeddings))
all_data = qa_data + reading_data

def retrieve_relevant_info(query, top_k=5): ####TOP K IS THE NUMBER OF CONTEXTS YOU WANT TO STORE/RETRIEVE
    query_embedding = embed_model.encode(query)
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [all_data[i] for i in top_indices]

def format_retrieved_info(info):
    formatted = []
    for item in info:
        if isinstance(item, dict):  # Q&A pair
            formatted.append(f"Q: {item['question']}\nA: {item['answer']}")
        else:  # Reading material chunk
            formatted.append(f"Context: {item}")
    return "\n\n".join(formatted)

def generate_response(query, custom_prompt, session_context):
    relevant_info = retrieve_relevant_info(query)
    context = format_retrieved_info(relevant_info)
    
    # Include session context in the prompt
    session_context_str = "\n".join(session_context)
    
    prompt = f"""{custom_prompt}

Here is some relevant information:

{context}

Previous context from this session:
{session_context_str}

Now, please answer the following question:
{query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.4)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

# Set default prompt
default_prompt = """You are an AI assistant knowledgeable about the Twelve Steps and Twelve Traditions of Alcoholics Anonymous. 
Your task is to provide accurate and helpful information based on the principles of AA. 
Be compassionate and supportive in your responses, while maintaining the integrity of AA's message."""

# Example usage
print("Welcome to LLaM_AA_AI!")
print(f"Current prompt: {default_prompt}")
print("You can change the prompt by typing 'change prompt' instead of a question.")

# Initialize session context
session_context = deque(maxlen=5)  # Keeps the last 5 Q&A pairs

while True:
    query = input("\nEnter your question (or 'change prompt' or 'quit'): ")
    if query.lower() == 'quit':
        break
    elif query.lower() == 'change prompt':
        default_prompt = input("Enter new prompt: ")
        print(f"Prompt updated. New prompt: {default_prompt}")
    else:
        print(f"\nQuestion: {query}")  # Print the query before generating the response
        response = generate_response(query, default_prompt, session_context)
        print(f"Response: {response}")
        
        # Add the current Q&A pair to the session context
        session_context.append(f"Q: {query}\nA: {response}")

print("Thank you for using LLaM_AA_AI!")
