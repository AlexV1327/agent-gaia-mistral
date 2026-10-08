# GAIA Tool Agent

Agent de questions-reponses experimente sur le benchmark GAIA. Le projet explore une approche d'agent outille: un modele Mistral orchestre des outils Python pour lire des fichiers, interroger des APIs, faire des recherches web et resoudre des taches multi-etapes.

J'ai construit ce projet pour comprendre concretement ce qui fait echouer ou reussir un agent sur des taches realistes: recuperer une source fiable, choisir le bon outil, parser un fichier, verifier un calcul et respecter un format de reponse strict. Le code a ete developpe par iterations, avec des runs d'evaluation et des analyses d'erreurs pour ameliorer les outils et l'orchestration.

## Objectif

GAIA contient des questions qui demandent souvent plus qu'une simple reponse textuelle: recherche web, lecture de PDF, tableurs, images, APIs externes, calculs, verification de sources et formatage strict. Ce projet sert a tester une architecture d'agent capable de combiner ces operations de maniere traceable.

## Fonctionnement

Le point d'entree principal est `gaia_runner.py`.

Pour chaque question, le runner:

1. charge une question GAIA depuis Hugging Face;
2. recupere l'eventuel fichier attache;
3. transmet la question a `GaiaAgent`;
4. enregistre la reponse, la trace des outils, la reponse attendue et le score dans un fichier JSONL.

L'agent peut utiliser notamment:

- recherche web;
- extraction HTML/PDF;
- lecture de fichiers texte, CSV, XLSX, PPTX et ZIP;
- analyse d'images locales via un modele vision Mistral configurable;
- calculs Python;
- APIs externes comme GitHub, Wikipedia, OpenReview, Wikidata ou USGS;
- solveurs specialises pour certaines familles de taches.

Une trace des outils appeles est conservee pour chaque question afin de comprendre pourquoi une reponse a ete produite et de diagnostiquer les echecs.

## Evaluation

L'agent est concu pour repondre sans connaitre les reponses a l'avance. Il combine recherche, lecture de fichiers, APIs et calculs, puis enregistre la trace des outils utilises pour permettre l'analyse des erreurs.

Run aleatoire:

```bash
.venv/bin/python gaia_runner.py --random --limit 20
```

Pour un run reproductible:

```bash
.venv/bin/python gaia_runner.py --random --limit 20 --seed 42
```

Pour tester un modele Mistral plus leger:

```bash
.venv/bin/python gaia_runner.py --random --limit 20 --seed 42 --model ministral-3b-latest
```

Exemple de tache isolee:

```bash
.venv/bin/python gaia_runner.py --start 62 --limit 1
```

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Renseigner ensuite les variables dans `.env`:

```bash
MISTRAL_API_KEY=...
HF_TOKEN=...
MISTRAL_MODEL=mistral-small-latest
MISTRAL_VISION_MODEL=pixtral-12b-latest
MISTRAL_OCR_MODEL=mistral-ocr-latest
```

## Utilisation

Run aleatoire:

```bash
.venv/bin/python gaia_runner.py --random --limit 20
```

Run aleatoire reproductible:

```bash
.venv/bin/python gaia_runner.py --random --limit 20 --seed 42
```

Run avec un modele Mistral specifique:

```bash
.venv/bin/python gaia_runner.py --random --limit 20 --seed 42 --model ministral-3b-latest
```

Run par index:

```bash
.venv/bin/python gaia_runner.py --start 75 --limit 5
```

## Analyse des runs

Les resultats sont sauvegardes dans `runs/*.jsonl` en local. Ces fichiers sont ignores par Git par defaut pour eviter de publier des artefacts volumineux ou accidentels.

Pour analyser un run:

```bash
.venv/bin/python scripts/analyze_runs.py runs/gaia_YYYYMMDD_HHMMSS.jsonl
```

L'analyse regroupe les echecs par type de question, type d'erreur et trace d'outils afin de guider les ameliorations.

Les erreurs temporaires d'API sont marquees dans les JSONL via `error_kind` (`rate_limit`, `timeout`, `temporary_api_error`, etc.) pour eviter de les confondre avec des erreurs de raisonnement de l'agent.

## Resultats observes

Les performances varient fortement selon le tirage des questions. Sur des lots de 20 questions, le projet a observe:

- de bons resultats sur certains lots lorsque les sources et fichiers sont bien exploitables;
- des variations importantes selon la proportion de questions visuelles, documentaires ou web.

Ces chiffres doivent etre lus comme des resultats experimentaux, pas comme une evaluation officielle du benchmark GAIA.

Exemples de familles de taches bien gerees:

- comptages Wikipedia via API MediaWiki;
- lecture de fichiers CSV, XLSX, PDF et PPTX;
- calculs bases sur donnees extraites;
- questions demandant de combiner une source web et un fichier attache.

Exemples de familles encore difficiles:

- images demandant une interpretation visuelle fine;
- videos ou pages web tres dynamiques;
- questions ou la source fiable est difficile a identifier automatiquement;
- enchainements longs ou une petite erreur de parsing change completement la reponse.

## Limites connues

- OCR et interpretation d'images encore fragiles;
- videos et pages web dynamiques difficiles a exploiter automatiquement;
- extraction PDF parfois instable selon la mise en page;
- dependance a des APIs externes et a leur disponibilite;
- certains outils restent tres specialises.

## Structure

```text
agent/
  llm.py       Client Mistral avec retry
  runner.py    Orchestration de l'agent et des outils
  tools.py     Outils de recherche, parsing, calcul et solveurs

gaia_runner.py            Runner principal GAIA
scripts/analyze_runs.py   Analyse des fichiers de resultats JSONL
requirements.txt          Dependances Python
.env.example              Exemple de configuration locale
```

## Ethique d'evaluation

Une partie importante du projet a consiste a eviter les reponses hardcodees: l'agent doit chercher, lire, calculer et verifier plutot que reconnaitre une question deja vue. Les traces d'outils servent a controler ce comportement et a comprendre les echecs.

Le projet a ete developpe avec assistance IA, mais les choix d'architecture, les runs et les analyses d'erreurs ont ete conduits manuellement.
