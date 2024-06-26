# Teambot

L'objectif est de créer un assistant apte à améliorer la productivité d'une équipe travaillant sur un projet.
Ceci est un logbook qui montre l'évolution de ce projet au fil du temps
**26/06/2024**
- Notre premier objectif est de rapatrier des données issues du web pour rendre le LLM plus expert dans unn domaine donné.
- Cela est possible avec le logiciel perplexity (payant dans sa version pro) mais comme nous utiliserons son équivalent opensource [perplexica](https://github.com/ItzCrazyKns/Perplexica)
  Nous avons installer Perplexica en suivant les instructions avec les API d'openAI, de Groq ainsi que Ollama. Ollama doit être installé via Docker:
  
  `docker pull ollama/ollama:latest`
  `docker run -d -p 11434:11434 ollama/ollama:latest`

  Nous avons un problème de connexion au serveur à résoudre quand on lance perplexica.
