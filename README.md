# York IE LLM Evaluation Demo

This demo emulates evaluation of the performance of several Large Language Models in a RAG-based Q&A system using [Arthur Bench](https://www.arthur.ai/product/bench).  

## Data
We consider a dataset of sample user questions about the [Autodromo Nazionale di Monza](https://en.wikipedia.org/wiki/Monza_Circuit), a historic motor racing circuit in Italy.
```json
{
    "user_query": "How often are Formula 1 cars at full throttle during a lap at Monza?",
    "context": "Monza, throughout its long and storied history has been known for its high-speed, simplistic nature (compared to \"harder\" circuits such as Singapore or Monaco, which are tight, unforgiving street circuits with lots of corners) thanks to its 1920s design and the few alterations it has received, and is currently the fastest track on the Formula One calendar and has been so since 1991. Monza consists of very long straights and tight chicanes, putting a premium on good braking stability and traction. The 5.793 km (3.600 mi)[3] circuit is very hard on engines; Formula 1 engines are at full throttle for nearly 80% of the lap, with engine failures common, notably Fernando Alonso in the 2006 Italian Grand Prix or Nico Rosberg in the 2015 Italian Grand Prix.",
    "response": "Formula 1 cars are at full throttle for over 80% of a lap at Monza."
}
```

- **user_query** (str):  
Sample user question
- **context** (str):  
 Chunk of text relevant to the `user_query`, retrieved from Wikipedia and used to augment the prompt fed to each LLM
- **response** (str):  
  "Golden" response to which LLM-generated responses will be compared

## Setup
### Permissions

**OpenAI**  

Generate an OpenAI API key with access to `gpt-4o-mini` and save to `/arthur-bench-demo/openai_api_key.txt`    
  

**AWS Bedrock**

[Request access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html) to the following AWS Bedrock models in region `us-west-2`:  

- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `meta.llama3-1-70b-instruct-v1:0`


### Environment
We recommend creating a python virtual environment to avoid dependency conflicts.   

**Via Conda**  

[Install anaconda](https://docs.anaconda.com/anaconda/install/)  

Navigate to project root 
```bash
cd /arthur-bench-demo 
```

Create environment via `env.yml`  
```bash
conda env create -f env.yml 
```

Activate environment
```bash
conda activate arthur-bench-demo
```  

**Via Python venv**  

[Install Python 3.10](https://www.python.org/downloads/)

Create environment 
```bash
python -m venv ${VENV_PATH}
```  

Activate environment
```bash
source ${VENV_PATH}/bin/activate
```  

Install requirements
```bash
pip install -r /arthur-bench-demo/requirements.txt
```

## Usage

All commands should be run from the working directory `/arthur-bench-demo/`

**Generate Candidate Responses and Create Test Suites**  

Run the provided python script
```bash
python compare_models.py
```

**Start Bench Webserver**  

Start Bench `uvicorn` webserver and note webserver address (usually `127.0.0.1:8000`)
```bash
bench
```

**Navigate to Bench console in web browser**  

Explore results by navigating to webserver address in browser, e.g. `http://127.0.0.1:8000`
