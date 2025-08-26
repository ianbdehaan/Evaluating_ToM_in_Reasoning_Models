from tqdm import tqdm
import pandas as pd


def psicExperiments(modelName, model, df):
    # file = open(outputFileName + ".txt", 'a')
    print(f'Initiating test for model {modelName} ->')
    for index in tqdm(df.index):
        O_1 = None
        while not O_1:
            try:
                O_1 = model.prompt(df.loc[index, 'Q_1'], df.loc[index, 'S'])
            except Exception as e:
                print(e)
        if df.notna().loc[index, 'Q_2']:
            O_2 = None
            while not O_2:
                try:
                    O_2 = model.reprompt(O_1, df.loc[index, 'Q_2'])
                except Exception as e:
                    print(e)
        else:
            O_2 = [('',''),]
        outputMap = {'A_1': O_1[0][0], 'R_1': O_1[0][1], 'A_2': O_2[0][0], 'R_2': O_2[0][1]}
        for column, value in outputMap.items():
            df.loc[index, column] = value
        df.to_csv(f'psic-results-{modelName}.csv', index=False)
    print(f'Results written to -> ./psic-results-{modelName}.csv')

def psicExperimentsNoReasoning(modelName, model, df):
    # file = open(outputFileName + ".txt", 'a')
    print(f'Initiating test for model {modelName} ->')
    for index in tqdm(df.index[54:]):
        O_1 = None
        while not O_1:
            try:
                O_1 = model.prompt(df.loc[index, 'Q_1'], df.loc[index, 'S'])
            except Exception as e:
                print(e)
        if df.notna().loc[index, 'Q_2']:
            O_2 = None
            while not O_2:
                try:
                    O_2 = model.reprompt(O_1, df.loc[index, 'Q_2'])
                except Exception as e:
                    print(e)
        else:
            O_2 = [('',''),]
        if df.notna().loc[index, 'Q_3']:
            O_3 = None
            while not O_3:
                try:
                    O_3 = model.reprompt(O_2, df.loc[index, 'Q_3'])
                except Exception as e:
                    print(e)
        else:
            O_3 = [('',''),]
        outputMap = {'A_1': O_1[0][0], 'A_2': O_2[0][0], 'A_3': O_3[0][0]}
        for column, value in outputMap.items():
            df.loc[index, column] = value
        df.to_csv(f'psic-results-no-reas-{modelName}.csv', index=False)
    print(f'Results written to -> ./psic-no-reas-results-{modelName}.csv')

def modExperiments(modelName, model, df):
    print(f'Initiating test for model {modelName} ->')
    for index in tqdm(df.index):
        out = None
        while not out:
            try:
                out = model.prompt(df.loc[index, 'Q'])
            except Exception as e:
                print(e)
        outputMap = {'A': out[0][0], 'R': out[0][1]}
        for column, value in outputMap.items():
            df.loc[index, column] = value
        df.to_csv(f'mod-results-{modelName}.csv', index=False)
    print(f'Results written to -> ./mod-results-{modelName}.csv')

def modExperimentsNoReasoning(modelName, model, df):
    print(f'Initiating test for model {modelName} ->')
    for index in tqdm(df.index):
        out = None
        while not out:
            try:
                out = model.prompt(df.loc[index, 'Q_1'])
            except Exception as e:
                print(e)
        out2 = None
        while not out2:
            try:
                out2 = model.reprompt(out, df.loc[index, 'Q_2'])
            except Exception as e:
                print(e)
        outputMap = {'A_1': out[0][0], 'A_2': out2[0][0]}
        for column, value in outputMap.items():
            df.loc[index, column] = value
        df.to_csv(f'mod-results-no-reas-{modelName}.csv', index=False)
    print(f'Results written to -> ./mod-results-no-reas-{modelName}.csv')
