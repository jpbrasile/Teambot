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
  - Image/vidéo to text (en particulier vidéos youtube)
  - Via internet (en particulier assimilation des codes disponibles sur Github)
    
- **Assimiler** les données :
  - Dans sa mémoire à court terme (contexte)
  - Dans sa mémoire à long terme (RAG)
  - Dans ses "gènes" (fine-tuning)
    
- **Activer** des ressources spécifiques (function calling)
  
- **Créer et utiliser des outils** soit disponible sur API (gorilla) soit qu'il crée lui même en les programmant
  
- **Créer des agents** susceptible de devenir expert dans un domaine donné grâce à sa capacité d'apprentissage et à la maîtrise d'outils appropriés 

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
  - Installation faie sur docker
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
