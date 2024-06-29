# Teambot

**Executive summary**
Pour améliorer la productivité, la première étape consiste à vérifier si notre besoin n'est pas déjà couvert par ailleurs et accessible sur le Web.
- Perplexity permet de faire ce type de recherche mais n'étant pas open source on ne peut automatiser sa mise en oeuvre et le traitement des données collectées.
- Perplexica est son équivalent open source que nous avons installé en local. Le logiciel doit néanmoins être adapté pour permettre le traitement des données collectées.
- Continue est un logiciel open source permettant de faire une telle adaptation
 


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

**27/06/2024**
- Il nous faut pouvoir modifier le programme Perplexica disponible sur Github
- [continue](https://github.com/continuedev/continue) peut analyser et adapter le programme

**28/06/2024**
... malheureusement _continue_ ne possède pas d'API. Nous avons trouvé une alternative: 

**Résumé du besoin :**

Vous souhaitez automatiser l'utilisation de la commande @codebase de l'assistant de codage Continue, afin d'analyser automatiquement un référentiel entier de code. Vous cherchez une solution pour intégrer cette fonctionnalité dans un programme Python.

**Proposition de solution :**

Bien que l'automatisation directe de @codebase via un programme Python ne soit pas une fonctionnalité standard de Continue, voici une approche alternative que vous pourriez envisager :
- Utilisez un script Python pour consolider tout le contenu du référentiel dans un seul fichier texte, en préservant la structure des chemins de fichiers.
- Employez un modèle de langage large (LLM) avec une grande fenêtre de contexte, comme DeepSeek Coder V2, qui peut traiter de grandes quantités de code.
- Créez une interface en Python pour interagir avec le LLM, lui permettant d'analyser le fichier consolidé et de répondre à des questions sur le code.
- Intégrez un agent de recherche web pour compléter les informations manquantes si nécessaire.
  
Cette approche vous permettrait d'obtenir une fonctionnalité similaire à @codebase, mais de manière automatisée et intégrée à votre programme Python. Cela vous donnerait la flexibilité d'analyser l'ensemble du référentiel et d'interagir avec le code de manière programmatique.

[Sonnet 3.5 donne le code](https://claude.ai/chat/aa8d87af-aa47-41ad-b364-d082ba649184)

Le fichier généré est trop important pour être exploité  par sonnet 3.5 et ChatGPTo. Nous téléchargeons deepseekcoder (contexte de 100 k) 

**29/06/2024**
- Abonnement à Antropic pro afin d'avoir un fonctionneemnt de sonnet 3.5 optimal (200 k de contexte).
- Répertoire TeambotV1 crée avec Perplexica fonctionnel 
- Nous avons relancé une adaptation de perplexica pour récupérer les données en local avec succès grâce à [sonnet 3.5](https://claude.ai/chat/5a6553cd-6040-459d-98c5-d37b1dc359a5).
