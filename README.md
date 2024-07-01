# Teambot

Nous allons entre**prendre** un voyage qui vous permettra de vous é**prendre** de l'Intelligence artificielle car nous allons créer votre nouvel ami intelligent et cultivé, toujours disponible et bienveillant à  votre égard. 
- Il va commencer par **prendre** toute sortes de données (texte, image, son, vidéo...)
- Il va devoir les com**prendre** afin d'utiliser ce savoir pour vous ap**prendre** tout ce que vous souhaitez.   
- Pour nos enseignants, il leur faudra désap**prendre** leurs méthodes traditionnelles comme l'apprentissage des langues ou de la programmation pour s'adapter à ce nouvel outil. 
- Mais il ne faut pas vous mé**prendre**, à charge à vous de re**prendre** le contrôle des opérations et nul doute que les résultats ne manqueront pas de vous sur**prendre** !!! 

# Teambot : Cahier des charges

## Introduction

Le travail d'équipe peut largement être amélioré avec l'aide des dernières technologies informatiques :

- Les « Large Language Models » (LLM) ont des capacités qui s'améliorent de jour en jour.
- Les GPU permettent des exécutions ultra-rapides de tâches complexes telles que les modélisations 3D.

Cependant, il n'existe pas actuellement d'environnement permettant un travail collaboratif efficace et simple d'emploi. **Teambot** vise à combler ce manque.

Ce document établit le cahier des charges de Teambot et identifie les frameworks les plus pertinents pour sa mise en œuvre.

[solution concurrente](https://abacus.ai/chat_llm-ent)
  - 🤖 Introduction de ChatLLM : une plateforme tout-en-un pour accéder à plusieurs modèles de langage avancés.
  - 📄 Capacité à télécharger et à analyser des PDF et autres documents pour générer des graphiques et des analyses.
  - 🔗 Intégration avec des services tiers comme Slack et Google Drive pour plus de commodité.
  - 💬 Création de chatbots personnalisés et d'agents IA déployables en production.
  - 🌐 Fonctionnalités de recherche web, génération d'images, exécution de code, et création de graphiques.
  - 📊 Utilisation de techniques de fine-tuning et de génération augmentée pour construire des chatbots adaptés à des bases de connaissances spécifiques.
  - 🧑‍💻 Équipe de développement composée d'ingénieurs logiciels, d'experts en apprentissage machine, et de consultants en IA pour fournir des solutions IA pour les entreprises et les usages personnels.
  - 🆓 Période d'essai gratuite d'un mois pour tester la plateforme avant de s'engager.

## Fonctionnalités principales

### 1. Workspace de travail

- Un espace dédié par projet, ouvert à l'équipe chargée de sa mise en œuvre.
- Supervision des échanges pour correspondre aux valeurs et objectifs de l'entreprise.

### 2. Conservation intelligente du projet

- Stockage des données brutes du projet.
- Possibilité d'interroger le projet en langage naturel pour obtenir des réponses adéquates.

### 3. LLM local et open source

- Contrôle complet du fonctionnement.
- Conservation de la confidentialité lorsque nécessaire.
- Utilisation d'un prompt système.

### 4. Mémoire à court terme importante

- Permet des échanges entre les divers acteurs sans perte de connaissance.

### 5. Bot dédié à chaque membre de l'équipe

Chaque bot personnel dispose de :

- Un espace personnel accessible uniquement par l'utilisateur (stockage des données sur son PC).
  - Sert d'assistant personnel et d'outil de formation continue.
- Accès aux données du projet pertinentes pour l'utilisateur.
- Accès à des outils spécifiques pour faciliter les tâches.
- Interface via l'écrit ou la parole.

#### Capacités du bot

Le bot doit pouvoir :

- **Prendre** des données :
  - Fichiers locaux
  - Speech-to-text
  - Image/vidéo to text (en particulier vidéos YouTube) :GPT-4o  devrait avoir cette capacité dans un avenir proche mais [des astuces sont possibles dès aujourd'hui](https://chatgpt.com/share/3afc20a8-ac01-410a-9288-0059c99780e9) 
  - Via internet (en particulier assimilation des codes disponibles sur Github)
    
- **Assimiler** les données :
  - Dans sa mémoire à court terme (contexte)
  - Dans sa mémoire à long terme (RAG)
  - Dans ses "gènes" (fine-tuning)
    
- **Activer** des ressources spécifiques (function calling)
  
- **Créer et utiliser des outils** soit disponible sur API (gorilla) soit qu'il crée lui même en créant le programme, en le lançant et en itérant suivant les erreurs rencontrées. 
  
- **Créer des agents** susceptibles de devenir experts dans un domaine donné grâce à leur capacité d'apprentissage et à leur maîtrise d'outils dédiés.

- **Exemple d'expertises utiles à Teambot**:
  - [Expert en RAG](Generative AI/local-rag-ollama-mistral-chroma.py)
  - Expert en modélisation Walk on Sphères pour l'optimisation d'objet 3D en électrostatique, magnétostatique, Navier/Stokes , thermique ...;
  - Expert en ModelingToolkit (ModelingToolkit.jl est un cadre de modélisation pour les calculs symboliques-numériques à haute performance en informatique scientifique et en apprentissage automatique scientifique.) 
  - Expert en Grasshopper pour la modélisation paramétrique d'objets complexes
  - Expert en pilotage de convertisseur par microprocesseur
  - Expert en jumeaux numériques
  - Expert en web scaping (comme perplexica)
  - [Expert en programmation](https://github.com/huangd1999/AgentCoder)
  - [Expert en optimisation inverse](https://github.com/AI4Science-WestlakeU/cindm)
  - [Expert en création d'agent en tant que service](https://github.com/run-llama/llama-agents?tab=readme-ov-file)
  - ...

## Principes fondamentaux

- Le bot est un outil permettant un travail plus efficace, mais **piloté par l'homme qui en assure le contrôle et la pertinence**.
- Objectif : éviter au mieux les hallucinations.

## Ressources matérielles

- Implantation locale de Teambot sur un serveur.
- Accessible à tout membre de l'équipe disposant des droits nécessaires, sans besoin de PC haute performance individuel.

## Frameworks de référence

S'inspirer des frameworks existants :
- Autogen
- CrewAI
- MemGPT
- AnythingLLM

## LLM adaptés aux besoins

### Assimilation rapide
- [Llama-3-8B-Instruct-Gradient-1048k](https://huggingface.co/spaces/Cyleux/Llama-3-8B-Instruct-Gradient-1048k) : Capable d'assimiler rapidement 1 million de tokens.
  - Nous l'avons installé sur Lmstudio. Il n'est pas très intelligent ni instruit ...
    
### Production de code
- [Codestral](https://mistral.ai/news/codestral/) : 32k contexte, 81.1 sur HumanEval.

### Meilleurs LLM actuels
- Via API : [Claude 3.5 Sonnet](https://apidog.com/blog/claude-3-5-api/)
    - Abonnement pris
- En local : [MoA (Mixture of Anthropic Models)](https://github.com/togethercomputer/MoA)

### Récupération de données sur le web
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) : Similaire à Perplexity.
  - Installation faite sur Docker
## Techniques avancées

### Fine-tuning
- [Guide de fine-tuning](https://www.perplexity.ai/search/How-to-finetune-sGJ9CD6zQ..8X.a9AsN_og)

### Function calling
- [Guide complet du function calling dans les LLM](https://thenewstack.io/a-comprehensive-guide-to-function-calling-in-llms/)


# Executive summary

L'objectif est de créer un assistant apte à améliorer la productivité d'une équipe travaillant sur un projet.

Pour améliorer la productivité, la première étape consiste à vérifier si notre besoin n'est pas déjà couvert par ailleurs et accessible sur le Web.
- Perplexity permet de faire ce type de recherche mais n'étant pas open source on ne peut automatiser sa mise en oeuvre et le traitement des données collectées.
- Perplexica est son équivalent open source que nous avons installé en local. Le logiciel a été adapté pour permettre le traitement des données collectées.
- Nous allons dans un premier temps utiliser "sonnet 3.5" qui est le meilleur LLM actuel. Nous avons souscrit à la version pro et à l'utilisation via API de ce logiciel.
 


# Logbook
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

**30/06/2024**
- Je me suis abonné à l'API antropic et j'ai réalisé mon premier chat "helloworld.py avec bien sûr sonnet 3.5 qui m'a donné le code correspondant !
- Nous pouvons maintenant récupérer les données sur n'importe quel sujet via le net et stocker ces informations localement. La taille de ces données peut dépasser le contexte, je vais donc réaliser un RAG avec l'aide de Sonnet.

- Récupération de données sur n'importe quel sujet via le web
- Stockage local des informations récupérées
- Utilisation de Sonnet pour créer une base de données questions-réponses
- Mise en place d'un système RAG (Retrieval Augmented Generation) avec cette base de données
- Recherche de corrélation entre la question posée et les questions stockées
- Possibilité d'utiliser la base de données pour l'alignement d'un LLM open source

Les principaux avantages de cette approche sont :
- Meilleure corrélation entre les questions et les réponses par rapport à un RAG traditionnel
- Base de connaissances personnalisée et spécifique au domaine d'intérêt
- Potentiel d'amélioration de la précision et de la pertinence des réponses
- Flexibilité pour mettre à jour et enrichir continuellement la base de données

Cependant, il faudra relever certains défis :
- Complexité technique dans la mise en œuvre du système Sonnet
- Assurance de la qualité des paires questions-réponses générées
- Gestion efficace de la base de données à mesure qu'elle s'agrandit
- Mise en place d'un système d'évaluation robuste pour mesurer l'efficacité

Cette approche innovante combine plusieurs technologies avancées (RAG, Sonnet, alignement de LLM) pour potentiellement créer un système plus performant et personnalisé.

**01/07/2024**
- **Web scraping :**
  - [00:00](https://www.youtube.com/watch?v=KAvuVUh0XU8&t=0s) 🌐 Crawl4AI is an open-source, LM-friendly web crawler and scraper that supports multiple URLs, extracts media tags, and returns structured data in JSON format.
  - [01:06](https://www.youtube.com/watch?v=KAvuVUh0XU8&t=66s) 📦 Using Crawl4AI simplifies web scraping by automating the process of defining elements, parsing data, and converting it into structured formats, integrated with AI agents.
  - [02:56](https://www.youtube.com/watch?v=KAvuVUh0XU8&t=176s) 🛠️ You can initiate a basic crawl and extract data from a URL using just a few lines of Python code with Coll 4 AI, demonstrating its ease of use and efficiency.
  - [04:48](https://www.youtube.com/watch?v=KAvuVUh0XU8&t=288s) 📊 Crawl4AI facilitates structured data extraction using LLM, allowing extraction of specific information like model names and pricing details from web pages.
  - [06:37](https://www.youtube.com/watch?v=KAvuVUh0XU8&t=397s) 🤖 Integrating Crawl4AI with AI agents such as web scraper, data cleaner, and data analyzer agents automates data extraction, cleaning, and analysis processes, generating detailed reports.
 
- **[Function calling LLM Benchmark](https://gorilla.cs.berkeley.edu/leaderboard.html)** : Gorilla est un très bon compromis open source et Sonnet 3.5 le meilleur à ce jour
- **Conversion de fichier au format Markdown :**
  -  [00:00](https://youtu.be/8446xEEq8RI?t=0s) 🛠️ Introduction à AutoMD

  - Présentation d'AutoMD, un outil Python pour convertir des fichiers en documents Markdown prêts pour LLM.
  - AutoMD est gratuit et fonctionne localement.

  - [01:23](https://youtu.be/8446xEEq8RI?t=83s) 📂 Fonctionnalités d'AutoMD
  
    - Supporte plusieurs types de fichiers et dossiers zip.
    - Génère des fichiers Markdown individuels ou multiples avec métadonnées et table des matières.
  
  - [02:16](https://youtu.be/8446xEEq8RI?t=136s) 📝 Formats de fichiers pris en charge
  
    - Supporte de nombreuses extensions de fichiers comme JSON, CSS, etc.
    - Mise à jour régulière des extensions supportées.
  
  - [03:25](https://youtu.be/8446xEEq8RI?t=205s) ⚙️ Installation d'AutoMD
  
    - Instructions pour installer AutoMD et créer un environnement Python.
    - Exemple de clonage et ouverture de projet dans VS Code.
  
  - [06:17](https://youtu.be/8446xEEq8RI?t=377s) 📁 Utilisation de l'interface utilisateur
  
    - Téléchargement de fichiers et sélection des options de sortie.
    - Processus de génération des fichiers Markdown avec table des matières et métadonnées.
  
  - [08:02](https://youtu.be/8446xEEq8RI?t=482s) 🔍 Conclusion et démonstration finale
  
    - Visualisation des fichiers générés avec les différentes métadonnées et contenu formaté.
    - Encouragement à tester l'outil et partage des retours.
   
  - **coding engineer**:
    - [00:00](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=0s) 🖥️ Introduction to Claude Engineer
      - Overview of Claude Engineer capabilities,
      - Describes how it assists in coding tasks, 
      - Example of creating a YouTube video downloader script.
    - [02:06](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=126s) 🛡️ Importance of Safety and Confirmation
      - Emphasis on the need for user confirmation in coding,
      - Discussion on safety measures to prevent unintended actions,
      - Mention of potential issues with agents and illegal activities.
    - [03:56](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=236s) 🎯 Enhancements and Future Projects
      - Demonstrates the flexibility of modifying scripts,
      - Transition to working on new projects like HTML and CSS,
      - Highlights of ongoing trends in AI tools and automation.
    - [05:06](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=306s) 🎮 Snake Game Implementation
      - Creation of a Snake game using Claude Engineer,
      - Explanation of the steps involved in setting up and running the game,
      - Insights into the capabilities of the tool in building functional applications.
    - [07:22](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=442s) 🌉 Advancements in AI Models
      - Discussion on the rapid progress of AI models,
      - Theory on how Anthropic improves model intelligence,
      - Reference to the Golden Gate Cloud experiment.
    - [10:08](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=608s) 🏃‍♂️ Competitive Progress of AI Companies
      - Comparison of Anthropic and OpenAI approaches,
      - Speculation on the future of AI model capabilities,
      - Reflection on the balance between user experience and model improvement.
    - [12:10](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=730s) 🚀 Exponential Improvement in AI Utility
      - Concept of users becoming more efficient with better AI tools,
      - Analogy of AI tools enhancing user capabilities like driving a better car,
      - Importance of adapting to and leveraging advanced AI technologies.
    - [14:56](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=896s) 🔧 Building and Using Advanced AI Tools
      - Example of winning a developer challenge with AI assistance,
      - Preview of upcoming live app projects,
      - Insights into the practical applications and future potential of AI tools.
    - [18:06](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1086s) 📑 Workflow and Development with Claude Engineer
      - Explanation of the workflow used to build Claude Engineer,
      - Demonstration of using Claude for function calls and documentation,
      - Step-by-step guide on starting a new project with Claude.
    - [20:31](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1231s) 📚 Importance of Training Data
      - Emphasizing the necessity of knowing what's in the training data,
      - Using documentation to ensure model accuracy.
    - [21:13](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1273s) 🔄 Best Practices for Function Calls
      - Describing function calling procedures,
      - Importance of running tools twice for verification.
    - [23:00](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1380s) 🧪 Testing and Experimentation
      - Creating and testing scripts quickly,
      - Demonstrating function calling with weather data.
    - [24:11](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1451s) 🚀 Encouraging Experimentation
      - Motivating viewers to start building projects,
      - Highlighting the ease of using AI tools for programming.
    - [24:38](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1478s) ❓ Community Engagement
      - Answering community questions,
      - Promoting community involvement in AI development.
    - [25:07](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1507s) 💡 Surprising Use Cases and Future Plans
      - Discussing unexpected use cases of Cloud Engineer,
      - Future functionalities and improvements.
    - [26:00](https://www.youtube.com/watch?v=tq9yN-j5o3M&t=1560s) 💼 Advice for AI Entrepreneurs
      - Encouraging solo entrepreneurs to build what resonates with them,
      - Importance of creating consumer-friendly AI tools.
     

     
    - [**Générateur de voix text to speech 2024**](https://www.youtube.com/watch?v=u5QnjiCRRmU):
      - 🎙️ Text to Speech Open AI est un outil de synthèse vocale gratuit de haute qualité.
      - 💸 Ce logiciel coûte seulement 6 $ pour une version payante, moins cher que d'autres outils similaires.
      - 📱 Il est accessible sur mobile et propose des voix réalistes avec des options d'émotion.
      - 🌐 L'interface est facile à utiliser : il suffit de chercher "text to speech open AI" sur Google.
      - 🎧 Chaque chaîne YouTube peut choisir une voix adaptée à son contenu, comme une voix motivante ou amusante.
      - 🖋️ Vous pouvez copier votre script, choisir la vitesse et la qualité audio, et générer jusqu'à 3000 mots gratuitement.
      - 🎶 Le logiciel Audacity peut être utilisé pour améliorer la qualité sonore de la voix générée.
      - 🗣️ L'outil permet également de créer des dialogues engageants entre plusieurs personnages.
    - [**Agent codeur**:](https://github.com/huangd1999/AgentCoder)
      - 🤖 Trois agents : AgentCoder utilise un agent programmeur, un agent concepteur de tests et un agent exécuteur de tests pour générer et tester du code.
      - 🌟 Performance supérieure : AgentCoder surpasse les modèles de LLM existants dans divers scénarios de codage.
      - 📈 Amélioration des résultats : AgentCoder augmente le pass@1 à 77.4% et 89.1% sur les ensembles de données HumanEval-ET et MBPP-ET.
      - 🔄 Format de sortie : Les agents suivent un format de sortie spécifique pour une analyse précise par l'agent exécuteur.
    - [**Création automatique d'agents**](https://github.com/jgravelle/AutoGroq)
      - 🤖 Introduction d'AutoGroq™ et son rôle dans la création d'agents IA.
      - 🖥️ Les agents IA sont des programmes informatiques autonomes.
      - 🚀 AutoGroq™ facilite la création d'agents IA pour les utilisateurs.
      - 🔄 Méthode standard vs. AutoGroq™ : résoudre d'abord le problème, puis créer l'agent spécialisé.
      - 🧩 Agents personnalisables : modification, ajout de compétences, apprentissage.
      - 🌐 Collaboration automatique des agents grâce à AutoGroq™ et autogen.
      - 🏗️ AutoGroq™ comme plateforme de construction et de test.
      - 🌍 Applications réelles et environnement de déploiement via autogen.
    - [**Monitoring des agents**](https://github.com/AgentOps-AI/agentops):
      - 🖥️ Présentation des défis des agents IA : coût, latence et observabilité.
      - 📊 Importance de la surveillance, des tests et des analyses pour les agents IA.
      - 🛠️ Configuration initiale et gestion des clés API pour AgentOps.
      - 🧩 Intégration de Crew AI avec AgentOps pour la surveillance des agents.
      - 📝 Développement du code pour initialiser et surveiller les agents IA.
      - 🔄 Définition des rôles et des tâches pour les agents Crew AI.
      - 🚀 Lancement et résultats de l'exécution des agents avec AgentOps.
      - 📢 Conclusion, encouragement à s'abonner et rejoindre la communauté Discord.    - [**Monitoring des agents**](https://github.com/AgentOps-AI/agentops):
      - 🖥️ Présentation des défis des agents IA : coût, latence et observabilité.
      - 📊 Importance de la surveillance, des tests et des analyses pour les agents IA.
      - 🛠️ Configuration initiale et gestion des clés API pour AgentOps.
      - 🧩 Intégration de Crew AI avec AgentOps pour la surveillance des agents.
      - 📝 Développement du code pour initialiser et surveiller les agents IA.
      - 🔄 Définition des rôles et des tâches pour les agents Crew AI.
      - 🚀 Lancement et résultats de l'exécution des agents avec AgentOps.
      - 📢 Conclusion, encouragement à s'abonner et rejoindre la communauté Discord.
    - [**Autogen update**](https://www.youtube.com/watch?v=ymz4RIUIask)
      - [00:00] 🧠 Microsoft AutoGen a reçu une mise à jour majeure pour les tâches complexes et l'amélioration des performances des agents.
      - [00:11] 🔧 AutoGen est un cadre de conversation multi-agent open source pour les applications de modèles de langage.
      - [00:40] 🚀 La mise à jour permet la collaboration entre agents pour accomplir des tâches multi-étapes plus efficacement que les solutions à agent unique.
      - [02:20] 💡 Adam Fourney de Microsoft a présenté cette amélioration en montrant comment les agents peuvent surpasser les solutions précédentes sur des benchmarks.
      - [02:59] 👥 Les agents peuvent se spécialiser et utiliser divers outils, permettant une meilleure génération pour des tâches complexes.
      - [05:00] 🔍 Exemple : résoudre des tâches complexes en utilisant une base de données, illustré par une recherche sur les crocodiles non indigènes en Floride.
      - [07:04] 🌐 AutoGen est open source et disponible sur GitHub.
      - [09:26] 📈 Les futurs développements incluent des agents capables d'apprendre et de s'améliorer, avec une meilleure compréhension des images et des captures d'écran.
     
   
- Extraction des données de Perplexica avec
  `(teambot) PS C:\Users\test\Documents\TeambotV1\temp_repo> python .\url-extractor-debug.py`. La requète à Perplexica est faite via http://localhost:3000/`
