""" Explore the Gemini models available for generating content. """

if __name__ == "__main__":
    import google.generativeai as genai
    from llama_index.llms.gemini import Gemini

    model_list = list(genai.list_models())
    supported_models = set()
    name_methods = {}

    for m in model_list:
        name_methods[m.name] = m.supported_generation_methods
        try:
            Gemini(model_name=m.name)
            supported_models.add(m.name)
        except Exception as e:
            name_methods[m.name] = f"{e}"

    supported_names = [m.name for m in model_list if m.name in supported_models]
    unsupported_names = [m.name for m in model_list if m.name not in supported_models]

    print(f"{len(model_list)} models: {len(supported_models)} supported models")
    print("Supported models: ---------")
    for i, name in enumerate(supported_names):
        print(f"{i:2}: {name:30} {name_methods[name]}")
    print("Unsupported models: ---------")
    for i, name in enumerate(unsupported_names):
        print(f"{i:2}: {name:30} {name_methods[name]}")

# 14 models: 9 supported models
# Supported models: ---------
#  0: models/gemini-1.0-pro          ['generateContent', 'countTokens']
#  1: models/gemini-1.0-pro-001      ['generateContent', 'countTokens', 'createTunedModel']
#  2: models/gemini-1.0-pro-latest   ['generateContent', 'countTokens']
#  3: models/gemini-1.0-pro-vision-latest ['generateContent', 'countTokens']
#  4: models/gemini-1.0-ultra-latest ['generateContent', 'countTokens']
#  5: models/gemini-1.5-pro-latest   ['generateContent', 'countTokens']
#  6: models/gemini-pro              ['generateContent', 'countTokens']
#  7: models/gemini-pro-vision       ['generateContent', 'countTokens']
#  8: models/gemini-ultra            ['generateContent', 'countTokens']
# Unsupported models: ---------
#  0: models/chat-bison-001          Model models/chat-bison-001 does not support content generation, only ['generateMessage', 'countMessageTokens'].
#  1: models/text-bison-001          Model models/text-bison-001 does not support content generation, only ['generateText', 'countTextTokens', 'createTunedTextModel'].
#  2: models/embedding-gecko-001     Model models/embedding-gecko-001 does not support content generation, only ['embedText', 'countTextTokens'].
#  3: models/embedding-001           Model models/embedding-001 does not support content generation, only ['embedContent'].
#  4: models/aqa                     Model models/aqa does not support content generation, only ['generateAnswer'].
