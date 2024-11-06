import arthur_bench.run.testsuite
import boto3
import botocore.config
import json
import jsonlines
import logging
import openai
import os
import pathlib
import pydantic
import sys
import traceback



## globals

ROOT = pathlib.Path(__file__).parents[::-1][-1]
DATA_FOLDER = ROOT / "data"
PROMPT_FOLDER = ROOT / "prompts"

with (ROOT / "openai_api_key.txt").open() as f_in:
    openai_api_key = f_in.read().strip()

openai.api_key = openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key


## class defs

class Record(pydantic.BaseModel):
    user_query: str
    context: str
    response: str

class FormattedHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stdout)
        fmt = '%(asctime)s %(name)s %(levelname)s: %(message)s'
        fmt_date = '%Y-%m-%dT%T'
        formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(formatter)

## fct defs

def get_logger(name: str, level: str | int = None):

    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    logger.addHandler(FormattedHandler())

    return logger


def load_jsonlines(p: pathlib.Path) -> list[dict[str, str]]:
    with jsonlines.open(p) as reader:
        d = list(reader)
    return d


def load_records(p: pathlib.Path) -> list[Record]:
    jsons = load_jsonlines(p)
    return list(map(Record.parse_obj, jsons))


def load_prompt(p: pathlib.Path) -> str:
    with p.open() as f_in:
        prompt = f_in.read()
    return prompt


def fill_user_template(record: Record) -> str:
    return USER_TEMPLATE.format(
        context=record.context.strip(),
        user_query=record.user_query.strip()
    )


def call_openai(record: Record, **kwargs) -> str:
    """
    Calls OpenAI chat completion endpoint given a Record record
    :param record:
    :param kwargs: passed to client.chat.completions.create
    :return: chat completion text
    """

    user_message = fill_user_template(record)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message}
    ]

    c = openai.ChatCompletion.create(messages=messages, **kwargs)

    return c.choices[0].message.content


def call_bedrock(client: boto3.client, model_id: str, body: str, **kwargs) -> dict:
    """
    :param client:
    :param model_id:
    :param body:
    :param kwargs: passed to client.invoke_model
    :return: chat completion text
    """

    r = client.invoke_model(
        modelId=model_id,
        body=body,
        accept="application/json",
        contentType="application/json",
        **kwargs
    )

    return json.loads(r['body'].read())


def call_claude(client: boto3.client, model_id: str, record: Record, model_kwargs: dict = None, **bedrock_kwargs) -> str:

    model_kwargs = model_kwargs or dict()

    """Anthropic suggests putting task-specific instructions in user turn"""
    system, task_instructions = SYSTEM_MESSAGE.split("\n", maxsplit=1)

    user_message = "\n\n".join([task_instructions.strip(), fill_user_template(record)])

    claude_body = json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'system': system.strip(),
        'messages': [{"role": "user", "content": user_message}],
        'max_tokens': 1_000,
        **model_kwargs
    })

    r = call_bedrock(client=client, model_id=model_id, body=claude_body, **bedrock_kwargs)

    return r['content'][0]['text']


def call_llama(client: boto3.client, model_id: str, record: Record, model_kwargs: dict = None, **bedrock_kwargs) -> str:

    model_kwargs = model_kwargs or dict()

    user_message = fill_user_template(record)
    llama_prompt = LLAMA_TEMPLATE.format(system_message=SYSTEM_MESSAGE, user_message=user_message)
    llama_body = json.dumps({'prompt': llama_prompt, **model_kwargs})

    r = call_bedrock(client=client, model_id=model_id, body=llama_body, **bedrock_kwargs)

    return r['generation']


SYSTEM_MESSAGE = load_prompt(PROMPT_FOLDER / "system.txt")
USER_TEMPLATE = load_prompt(PROMPT_FOLDER / "user.txt")
LLAMA_TEMPLATE = load_prompt(PROMPT_FOLDER / "llama_template.txt")

def main() -> bool:

    logger = get_logger(pathlib.Path(__file__).name)

    try:

        logger.info(f"Loading input data")

        input_records = load_records(DATA_FOLDER / "inputs.jsonl")
        bedrock = boto3.client('bedrock-runtime', config=botocore.config.Config(region_name="us-west-2"))

        input_text_list = list()
        reference_output_list = list()
        context_list = list()

        claude_summaries = list()
        llama_summaries = list()
        openai_summaries = list()

        logger.info(f"Generating candidate responses")

        for record in input_records:
            input_text_list.append(record.user_query)
            reference_output_list.append(record.response)
            context_list.append(record.context)

            claude_summaries.append(call_claude(bedrock, "anthropic.claude-3-5-sonnet-20241022-v2:0", record))
            llama_summaries.append(call_llama(bedrock, "meta.llama3-1-70b-instruct-v1:0", record))
            openai_summaries.append(call_openai(record, model="gpt-4o-mini"))

        logger.info("Initializing test suites")

        test_suite_qa = arthur_bench.run.testsuite.TestSuite(
            "Monza Trivia - QA Correctness",
            "qa_correctness",
            input_text_list=input_text_list
        )

        test_suite_bertscore = arthur_bench.run.testsuite.TestSuite(
            "Monza Trivia - Bertscore",
            "bertscore",
            input_text_list=input_text_list,
            reference_output_list=reference_output_list
        )


        logger.info(f"Running QA Tests")


        test_suite_qa.run("claude_summaries", candidate_output_list=claude_summaries, context_list=context_list)
        test_suite_qa.run("llama_summaries", candidate_output_list=llama_summaries, context_list=context_list)
        test_suite_qa.run("openai_summaries", candidate_output_list=openai_summaries, context_list=context_list)

        logger.info(f"Running BERTscore Tests")

        test_suite_bertscore.run("claude_summaries", candidate_output_list=claude_summaries)
        test_suite_bertscore.run("llama_summaries", candidate_output_list=llama_summaries)
        test_suite_bertscore.run("openai_summaries", candidate_output_list=openai_summaries)

        logger.info(f"Done. Start ArthurBench console server by running `bench` in working directory `{ROOT}`. Open ArthurBench console in browser to view results.")

    except:
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":
    sys.exit(not main())
