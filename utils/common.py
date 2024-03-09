import re
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, HuggingFacePipeline
from langchain.chains import LLMChain


SHP_TEMPLATE = """
You will be given a Reddit post and a reply. Your job is to judge whether the reply {attribute_desc}. Score that on a scale from 1 to 10 where 1 means {attr_min} and 10 means {attr_max}.

POST:
{history}

Reply:
{reply}

Answer by outputting a number from 1 to 10 (and nothing else).

Answer:"""
HH_TEMPLATE = """
You will be given a conversation between a human and an AI assistant. Your job is to judge whether assistant's reply {attribute_desc}. Score that on a scale from 1 to 10 where 1 means {attr_min} and 10 means {attr_max}. Here's the conversation:

Human:
{history}

Assistant:
{reply}

Answer by outputting a number from 1 to 10 (and nothing else).

Answer:"""
def get_lm(model_name, model_type):
    if model_type == 'openai':
        lm = ChatOpenAI(model_name=model_name, temperature=0)
    elif model_type == 'huggingface':
        lm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text2text-generation", model_kwargs={"do_sample":False},device=0)
    return lm

def feature_score(feature_extractor, **feature_kwargs):
    try:
        lm_response = feature_extractor(feature_kwargs)['score']
        score = re.search(r'^\d+', lm_response).group(0)
        if score:
            return int(score)
        else:
            return None
    except Exception as e:
        print(e)
        return None

def get_feature_extractor(lm,data_type):
    if data_type=='shp':
        feature_extractor_template = SHP_TEMPLATE
    elif data_type=='hh':
        feature_extractor_template = HH_TEMPLATE
    feature_extractor_prompt = PromptTemplate(
        input_variables=["history", "reply", "attribute_desc", "attr_min", "attr_max"],
        template=feature_extractor_template,
    )
    feature_extractor = LLMChain(llm=lm, prompt=feature_extractor_prompt, output_key="score")
    return feature_extractor
