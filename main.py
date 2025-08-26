from src.prompt_models import *
from src.experiment import psicExperiments, modExperiments, psicExperimentsNoReasoning, modExperimentsNoReasoning
import pandas as pd

models={
    'r1': DeepseekModel(),
    'gpt-5-high': OpenAIModel('gpt-5-2025-08-07', 'high'),
    'gpt-5-minimal': OpenAIModel('gpt-5-2025-08-07', 'minimal'),
    'gemini': GoogleGenAI(),
    'claude': AnthropicAI(),
    'claude-no-thinking': AnthropicAI(False),
    'grok-4': xAI('grok-4-0709'),
    'grok-3-mini': xAI('grok-3-mini')
}

experiments={
    'Psicological tests': lambda modelName, model: psicExperiments(modelName, model, pd.read_csv('data/queries_treated.csv', dtype=str)),
    'Psicological tests (no reasoning)': lambda modelName, model: psicExperimentsNoReasoning(modelName, model, pd.read_csv('data/queries_non_reasoning.csv', dtype=str)),
    'Simple Modifications': lambda modelName, model: modExperiments(modelName, model, pd.read_csv('data/modifications.csv', dtype=str)),
    'Modifications (no reasoning)': lambda modelName, model: modExperimentsNoReasoning(modelName, model, pd.read_csv('data/modifications_no_reasoning.csv', dtype=str))
}

def selection(category, dic):
    names = list(dic.keys())
    print(f'Select the {category} for the experiment:')
    for (index, key) in enumerate(names):
        print(f'  {index}) {key}')
    keyIdx=int(input('Enter a number:'))
    key=names[keyIdx]
    return key

def main():
    experimentName = selection('experiment',experiments)
    modelName = selection('model',models)
    experiments[experimentName](modelName, models[modelName])

if __name__ == '__main__':
    main()

