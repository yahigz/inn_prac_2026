import os

import openai


class BaseCallAPI():
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base
        self.api_key = api_key
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        self.model_name = model_name

    def load_prompt(self, file_dir):
        with open(file_dir, 'r') as file:
            prompt = file.read()
        return prompt

    def call_api(self, prompt, temperature):
        pass


class GPTCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super(GPTCallAPI, self).__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file, temperature=0.2):
        with open(prompt_file, 'r') as file:
            prompt = file.read()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            # model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a chatbot", },
                {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True
        )
        result = ""
        try:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content
                    # print(chunk.choices[0].delta.content, end="")
        except:
            pass

        return result


class LocalCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super(LocalCallAPI, self).__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file,
                 temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"

        with open(prompt_file, 'r') as file:
            prompt = file.read()

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
                stop=stop_tokens)
        return response["choices"][0]["message"]["content"]


def get_llm_api(args):
    model_name = str(getattr(args, 'llm_model', '') or '')
    api_base = str(getattr(args, 'api_base', '') or '').strip()
    api_key = str(getattr(args, 'api_key', '') or '').strip()

    if (not api_base) and model_name.startswith('openai/'):
        api_base = os.getenv('AUTOSAT_API_BASE') or os.getenv('DEEPINFRA_API_BASE') or ''

    if api_base:
        return GPTCallAPI(api_base=api_base,
                          api_key=api_key,
                          model_name=model_name)

    if model_name in ('gpt-4-1106-preview', 'gpt-3.5-turbo'):
        return GPTCallAPI(api_base=api_base,
                          api_key=api_key,
                          model_name=model_name)
    if model_name == 'Qwen':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/qwen/Qwen-72B-Chat")
    if model_name == 'llama':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    if model_name == 'deepseek':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/deepseek-ai/deepseek-coder-33b-instruct")

    raise NotImplementedError(
        "Unsupported llm_model without external API endpoint. "
        "Set api_base/api_key for OpenAI-compatible models or use one of: Qwen, llama, deepseek."
    )


if __name__ == '__main__':
    llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                          api_key="sk-",
                          model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    answer = llm_api.call_api(prompt_file='../template/EasySAT/bump_var_function/original_prompt.txt')
    print(answer)
