# Teambot

L'objectif est de créer un assistant apte à améliorer la productivité d'une équipe travaillant sur un projet.
Ceci est un logbook qui montre l'évolution de ce projet au fil du temps
**26/06/2024**
- Notre premier objectif est de rapatrier des données issues du web pour rendre le LLM plus expert dans unn domaine donné.
- Cela est possible avec le logiciel perplexity (payant dans sa version pro) mais comme nous utiliserons son équivalent opensource [perplexica](https://github.com/ItzCrazyKns/Perplexica)
  Nous avons installer Perplexica en suivant les instructions avec les API d'openAI, de Groq ainsi que Ollama. Ollama doit être installé via Docker:
  
  `docker pull ollama/ollama:latest`
  `docker run -d -p 11434:11434 ollama/ollama:latest`

  Nous avons un problème de connexion au serveur à résoudre quand on lance perplexica.L'erreur provenait de l'absence de crédit sur openai. J'utilse Groq llama70b moins cher et plus rapide à mettre dans les paramètres de Perplexica. L'IHM est analgue à celui de Perplexity mais fonctionne en local, les requètes web sont anonymisées.

  ![image](https://github.com/jpbrasile/Teambot/assets/8331027/bde4c7f5-dc9a-4c86-a3bc-9d475a334f74)

**Bibliographie:**

[llm réduit ultra-rapide](https://huggingface.co/PowerInfer/TurboSparse-Mixtral): Une nouvelle méthode de raréfaction basée sur dReLU qui augmente la parcimonie du modèle à 90 % tout en maintenant les performances, atteignant une accélération de 2 à 5 fois lors de l'inférence.

[agents s'améliorant avec le temps](https://arxiv.org/abs/2404.11964)

[Les agents intelligents serverless permettent d'automatiser et de gérer facilement des applications cloud sans avoir à s'occuper des serveurs](https://github.com/ruvnet/agileagents)

[🔧 Maestro est un cadre pour orchestrer intelligemment les sous-agents utilisant Claude Opus et d'autres modèles AI.
🔄 Il supporte plusieurs modèles AI comme Opus, Haiku, GPT-4o, LMStudio, et Ollama.
📦 Les scripts sont organisés en fichiers distincts pour chaque fonctionnalité AI.
🌐 Intégration d'une application Flask pour une interface utilisateur conviviale.](https://github.com/Doriandarko/maestro)

[Avec Sonnet 3.5 code avec web search and file management](https://github.com/Doriandarko/claude-engineer)

[Mille pages de data en mémoire court terme (contexte 1 million tokens)[https://huggingface.co/spaces/Cyleux/Llama-3-8B-Instruct-Gradient-1048k)

[ $0.03 per hour of transcription](https://console.groq.com/playground?model=whisper-large-v3)

[Open Web UI](https://github.com/open-webui/open-webui) offre une interface utilisateur conviviale et extensible pour gérer des modèles de langage (LLM) en local, compatible avec les API d'OpenAI et Ollama. Il propose des fonctionnalités avancées telles que la prise en charge des plugins Python, la communication vocale/vidéo, et la génération d'images, tout en étant accessible sur divers appareils.

[**MOOC pour se former aux agents (Autogen)**](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/)

[Base](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/2/multi-agent-conversation-and-stand-up-comedy)
