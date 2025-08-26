import pandas as pd
import sys
import os

def print_result(index, prompt):
    print(f'''
{index} - {prompt.Type} ->

- System:
{prompt.S}

- Question 1:
{prompt.Q_1}

- Reasoning 1:
{prompt.R_1}

- Answer 1:
{prompt.A_1}

- Correct Answer 1:
{prompt.CA_1}

- Question 2:
{prompt.Q_2}

- Reasoning 2:
{prompt.R_2}

- Answer 2:
{prompt.A_2}

- Correct Answer 2:
{prompt.CA_2}''')

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
