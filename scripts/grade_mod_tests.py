import pandas as pd
import sys
import os

def print_result(index, prompt):
    print(f'''
- Question:
{prompt.Q}

- Reasoning:
{prompt.R}

- Answer:
{prompt.A}

- Correct Answer:
{prompt.CA}''')

if __name__ == '__main__':
    try:
        inputFileName = sys.argv[1]
        outputFileName = sys.argv[2]
    except IndexError:
        print('Provide the file to be evaluated and the output file as command-line arguments')
        print('usage: python evaluate_correctness.py [input_file].csv [output_file].csv')
    df = pd.read_csv(inputFileName).fillna('')
    df['C'] = pd.NA
    df['C'].astype('boolean', copy = False)
    validAnswers = ['0','1','2']   

    for index, prompt in zip(df.index, df.iloc):
        os.system('clear')
        print_result(index, prompt)
        grade = None
        while grade is None:
            inp = input('Grade this response:\n')
            if inp in validAnswers:
               grade = int(inp) 
        df.loc[index, 'C'] = grade
        os.system('clear')

    df.to_csv(outputFileName, index = False)
