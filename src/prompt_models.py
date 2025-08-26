from openai import OpenAI
from google import genai
from google.genai import types
import anthropic


class DeepseekModel:
    def __init__(self):
        self.model = 'deepseek-reasoner'
        self.client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    def prompt(self, userPrompt, systemPrompt = None):
        prompt=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": userPrompt},
        ] if systemPrompt else [
            {"role": "user", "content": userPrompt},
        ]
        response = self.client.chat.completions.create(
            model = self.model,
            messages=prompt,
            stream=False,
            temperature=0
        )
        result = response.choices[0].message
        return (result.content, result.reasoning_content), prompt

    def reprompt(self, last, newUserPrompt):
        prompt = last[1]
        response = last[0][0]
        prompt.append({"role": "assistant", 'content': response})
        prompt.append({"role": "user", "content": newUserPrompt})
        response = self.client.chat.completions.create(
            model = self.model,
            messages=prompt,
            stream=False,
            temperature=0
        )
        result = response.choices[0].message
        return (result.content, result.reasoning_content), prompt
        

class OpenAIModel:
    def __init__(self, modelname, effort):
        self.model = modelname
        self.effort = effort
        self.client = OpenAI(api_key="")

    def prompt(self, userPrompt, systemPrompt = None):
        response = self.client.responses.create(
            model = self.model,
            reasoning={
                'effort': 'high',
                'summary': 'auto'
            },
            input= [
                {"role": "developer","content": [{"type": "input_text","text": systemPrompt}]},
                {"role": "user","content": [{"type": "input_text","text": userPrompt}]}
            ] if systemPrompt else [
                {"role": "user","content": [{"type": "input_text","text": userPrompt}]}              
            ],
            text={"format": {"type": "text"}},
            tools=[],
            store=True,
        )        
        summaries_list = response.output[0].summary
        summary =  '\n\n'.join([summaries_list[i].text for i in range(len(summaries_list))])
        return (response.output_text, summary), response.id

    def reprompt(self, last, newUserPrompt):
        response_id = last[1]
        response = self.client.responses.create(
            model = self.model,
            previous_response_id = response_id,
            reasoning={
                'effort': 'high',
                'summary': 'auto'
            },
            input= [
                {"role": "user","content": [{"type": "input_text","text": newUserPrompt}]}
            ],
            text={"format": {"type": "text"}},
            tools=[],
            store=True
        )        
        summaries_list = response.output[0].summary
        summary =  '\n\n'.join([summaries_list[i].text for i in range(len(summaries_list))])
        return (response.output_text, summary), response.id
   
    def interact(self, prompt):
        return self.prompt('',prompt)[0][0]


class GoogleGenAI:
    def __init__(self):
        self.client = genai.Client(api_key='')
        self.model = 'gemini-2.5-pro'

    def prompt(self, userPrompt, systemPrompt = None):
        prompt = [types.UserContent(parts=[types.Part.from_text(text=userPrompt)])]
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=systemPrompt if systemPrompt else None,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
                temperature = 0),
            contents=prompt,
        )
        return GoogleGenAI.answerAndThinking(response.candidates[0].content.parts), (prompt, systemPrompt)

    def reprompt(self, last, newUserPrompt):
        prompt, systemPrompt = last[1][0], last[1][1]
        modelResponse = last[0][0]
        prompt.append(types.ModelContent(parts=[types.Part.from_text(text=modelResponse)]))
        prompt.append(types.UserContent(parts=[types.Part.from_text(text=newUserPrompt)]))
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=systemPrompt if systemPrompt else None,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
                temperature=0),
            contents=prompt
        )
        return GoogleGenAI.answerAndThinking(response.candidates[0].content.parts), (prompt, systemPrompt)

    @staticmethod
    def answerAndThinking(parts):
        thinking=''
        answer=''
        for part in parts:
            if not part.text:
                continue
            if part.thought:
                thinking+=part.text
            else:
                answer+=part.text
        return answer, thinking

class AnthropicAI:
    def __init__(self, thinking = True):
        self.model="claude-opus-4-1-20250805"
        self.client = anthropic.Anthropic(api_key='', timeout=None)
        self.thinking = thinking

    def prompt(self, userPrompt, systemPrompt = None):
        prompt=[{"role": "user", "content": userPrompt}]
        message = self.client.messages.create(
            max_tokens=5000,
            thinking={
                "type": "enabled",
                "budget_tokens": 4000
            } if self.thinking else {"type": 'disabled'},
            model=self.model,
            system=systemPrompt if systemPrompt else '',
            messages=prompt
        )
        return AnthropicAI.answerAndThinking(message), (prompt, systemPrompt, message.content)

    def reprompt(self, last, newUserPrompt):
        prompt, systemPrompt = last[1][0], last[1][1]
        modelResponse = last[1][2]
        prompt.append({"role": "assistant", "content": modelResponse})
        prompt.append({"role": "user", "content": newUserPrompt})
        message = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            } if self.thinking else {"type": 'disabled'},
            system=systemPrompt if systemPrompt else '',
            messages = prompt
        )
        return  AnthropicAI.answerAndThinking(message), (prompt, systemPrompt, message.content)

    def interact(self, prompt):
        return self.prompt('',prompt)[0][0]
        
    @staticmethod
    def answerAndThinking(message):
        thinking=''
        answer=''
        for block in message.content:
            if block.type == "thinking":
                thinking+=block.thinking
            elif block.type == 'text':
                answer+=block.text
        return answer, thinking

class xAI:
    def __init__(self, modelName):
        self.model=modelName
        self.client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=""
        )

    def prompt(self, userPrompt, systemPrompt = None):
        prompt=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": userPrompt},
        ] if systemPrompt else [
            {"role": "user", "content": userPrompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=0
        )

        # reasoning is in completion.choices[0].message.reasoning_content, but not available in grok4
        message = completion.choices[0].message
        return ((message.content, message.reasoning_content), prompt) if self.model=='grok-3-mini' else ((message.content,''), prompt)

    def reprompt(self, last, newUserPrompt):
        prompt = last[1]
        modelResponse = last[0][0]
        prompt.append({"role": "assistant", "content": modelResponse})
        prompt.append({"role": "user", "content": newUserPrompt})
        message = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=0
        )
        return ((message.content, message.reasoning_content), prompt) if self.model=='grok-3-mini' else ((message.content,''), prompt)
        
    def interact(self, prompt):
        return self.prompt('',prompt)[0][0]
