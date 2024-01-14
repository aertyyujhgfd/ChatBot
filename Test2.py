import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
from bs4 import BeautifulSoup
from collections import deque

# Mécanisme de suivi du contexte
context = deque(maxlen=10)  # Limite la taille du contexte à 10 éléments

# Définir des préférences factices pour quelques utilisateurs
preferences = {
    "user123": {"langue": "fr", "thème": "clair"},
    "user456": {"langue": "en", "thème": "sombre"},
    # Ajoutez d'autres utilisateurs avec leurs préférences
}

def personalize_reply(reply, user_preferences):
    # Logique pour personnaliser la réponse en fonction des préférences de l'utilisateur
    # Vous devez implémenter cette fonction en fonction de vos besoins
    # Pour l'instant, cela renvoie simplement la réponse telle quelle
    return reply

def search_web(query):
    try:
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', {'class': 'tF2Cxc'})  # Ajustez la classe en fonction de la structure réelle
        results = [result.find('h3').get_text() for result in search_results]
        return results
    except Exception as e:
        print(f"Erreur lors de la recherche sur le web : {e}")
        return []

def get_user_id():
    # Implémentation factice, remplacez-la par la logique réelle pour obtenir l'identifiant de l'utilisateur
    return "user123"

def generate_response_t5(prompt, model, tokenizer, max_length=150):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    # Ajouter une logique pour gérer les réponses incomplètes
    if not reply.strip():
        reply = "Je ne suis pas sûr de comprendre. Pourriez-vous reformuler votre question ?"

    return reply

def chatbot(message, user_id):
    global context

    # Mécanisme de suivi du contexte
    context.append((user_id, message))

    # Utiliser T5 pour générer une réponse basée sur les résultats de recherche
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)

    # Concaténer les titres des résultats de recherche dans le prompt
    search_results = search_web(message)
    search_prompt = " ".join(result for result in search_results)
    
    # Encoder le prompt avec le contexte de manière plus explicite
    context_messages = [f"{user}: {msg}" for user, msg in context]
    context_prompt = " | ".join(context_messages)
    input_prompt = f"{context_prompt} | Vous: {message} | Recherche: {search_prompt}"

    # Générer la réponse avec T5
    reply = generate_response_t5(input_prompt, model, tokenizer, max_length=300)  # Ajustement de la longueur maximale

    # Logique de filtrage et protection des données
    if "mot_sensitive" in reply:
        reply = "Désolé, je ne peux pas fournir d'informations sensibles."

    # Logique de personnalisation des réponses
    if user_id in preferences:
        # Appliquer la personnalisation en fonction des préférences de l'utilisateur
        reply = personalize_reply(reply, preferences[user_id])

    # Ajouter une logique pour gérer les réponses vides ou inappropriées
    if not reply.strip():
        reply = "Je ne suis pas sûr de comprendre. Pourriez-vous reformuler votre question ?"

    # Afficher les résultats de la recherche
    print("Résultats de la recherche:", search_results)

    return reply

# Boucle principale du chatbot
while True:
    # Lire l'entrée de l'utilisateur et son identifiant
    user_input = input("Vous: ")
    user_id = get_user_id()  # Fonction factice pour obtenir l'identifiant de l'utilisateur

    # Vérifier si l'utilisateur veut quitter
    if user_input.lower() == 'exit':
        print("Chatbot: Au revoir !")
        break

    # Appel du chatbot avec l'entrée de l'utilisateur et son identifiant
    bot_reply = chatbot(user_input, user_id)

    # Afficher la réponse du chatbot
    print("Chatbot:", bot_reply)



