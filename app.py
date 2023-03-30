from transformers import pipeline, set_seed, Pipeline
import gradio as gr
import random
import re
from llama_index import GPTSimpleVectorIndex, download_loader
import os
from dotenv import load_dotenv

load_dotenv()

title = "Stable Diffusion Prompt Generator"


class App:

    gpt2_pipe: Pipeline

    def __init__(self):
        self.gpt2_pipe = pipeline(
            'text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')

    def get_keywords(self, url: str):
        BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
        loader = BeautifulSoupWebReader()
        documents = loader.load_data(urls=[url])
        index = GPTSimpleVectorIndex.from_documents(documents)
        response = index.query(
            use_async=False,
            query_str='Give me keywords of this page, return with comma delimiter and without title')
        if response != None:
            return str(response).strip() # type: ignore

    def generate(self, starting_text: str):
        print(starting_text)
        seed = random.randint(100, 1000000)
        set_seed(seed)

        if starting_text == "":
            starting_text = re.sub(r"[,:\-–.!;?_]", '', starting_text)

        response = self.gpt2_pipe(starting_text, max_length=(
            len(starting_text) + random.randint(60, 90)), num_return_sequences=4)

        response_list: list[str] = []
        for x in response:  # type: ignore
            resp = x['generated_text'].strip()  # type: ignore
            if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                resp = re.sub('[^ ]+\.[^ ]+', '', resp)
                resp.replace("<", "").replace(">", "")
                response_list.append(resp)
        return list(map(lambda item: [item], response_list))

    def launch(self):
        with gr.Blocks(title=title) as block:
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(lines=1, label="URL",
                                           placeholder="URL", interactive=True)
                    submit = gr.Button(
                        variant="primary", value="Generate")

                    keywords = gr.Textbox(
                        lines=1, label="Keywords", placeholder="Keywords", interactive=True)

                    change_btn = gr.Button(
                        variant="primary", value="Change keywords")

                with gr.Column():
                    def handle_select(x: gr.SelectData):
                        return x.value

                    dataframe = gr.DataFrame(
                        headers=['Prompt'],
                        datatype=['str'],
                        row_count=5,
                        col_count=(1, "fixed"),
                        wrap=True,
                    )

                    selected = gr.Textbox(label="selected", interactive=True)

                    def handle_keyword_change(x: str):
                        return self.generate(x)

                    def get_keywords(x: str):
                        keywords_resp = self.get_keywords(x)
                        if keywords_resp != None:
                            prompt = self.generate(keywords_resp)
                            return (keywords_resp, prompt)

            dataframe.select(handle_select, outputs=selected)
            submit.click(get_keywords, outputs=[
                         keywords, dataframe], inputs=url_input)
            change_btn.click(handle_keyword_change,
                             inputs=keywords, outputs=dataframe)

        block.launch(share=True)


app = App()

app.launch()
