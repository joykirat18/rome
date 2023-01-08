import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Tokenizer

# from modeling_utils import PreTrainedModel

torch.set_grad_enabled(False)




prompt = "8 + 8 ="
allHiddenValue = []
def model_hidden_state(mt, prompt):
    inp = make_inputs(mt.tokenizer, prompt)
    outputs_exp = mt.model(**inp, output_hidden_states=True, output_attentions=True)
    # print(outputs_exp.shape)
    out = outputs_exp["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    result = [mt.tokenizer.decode(c) for c in preds]
    print(result)
    return outputs_exp

model_name = "facebook/bart-base"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neo-2.7B" or "gpt2-xl" or "facebook/bart-base"

if("bart" in model_name):
    from bart import ModelAndTokenizer
    import bart
    mt = ModelAndTokenizer(model_name, torch_dtype=(torch.float16 if "20b" in model_name else None))
    base_model = model_hidden_state(mt, [prompt])
    allHiddenValue = bart.allHiddenValue     



from datetime import datetime
base_directory = str(datetime.now())
print(base_directory)
isExist = os.path.exists(base_directory)
if not isExist:
    os.makedirs(base_directory)
directory = base_directory
d = {'prompt':prompt,'model':model_name}
with open(directory + '/detail.txt', 'w') as f:
    f.write('dict = ' + str(d) + '\n')
import pickle
with open(directory  + '/hiddenValue.pickle', 'wb') as handle:
    pickle.dump(allHiddenValue, handle)
symbolResidual = []
symbol = prompt.split()
print(symbol)

for i in range(len(symbol)):
  values = []
  for residue in allHiddenValue:
    first = residue['original'].cpu().detach().numpy()[0][i].mean()
    second = residue['original+attn'].cpu().detach().numpy()[0][i].mean()
    third = residue['original+attn+feed'].cpu().detach().numpy()[0][i].mean()
    values.append(first)
    values.append(second)
    # values.append(third)
  symbolResidual.append(values)
    


import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams["figure.figsize"] = [7.50, 5.50]
plt.rcParams["figure.autolayout"] = True
for i in range(len(symbol)):
  y = np.array(symbolResidual[i])
  plt.plot(y, label = symbol[i])
# y2 = np.array([6, 2, 7, 11])
leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(prompt)

plt.savefig(directory + '/Mean_of_residual_Values')
# plt.show()

ratioResidual = []

for i in range(len(symbol)):
  values = []
  for residue in allHiddenValue:
    first = np.linalg.norm(residue['original'].cpu().detach().numpy()[0][i])
    attn = np.linalg.norm(residue['attn'].cpu().detach().numpy()[0][i])
    second = np.linalg.norm(residue['original+attn'].cpu().detach().numpy()[0][i])
    feed = np.linalg.norm(residue['feed_forward'].cpu().detach().numpy()[0][i])
    # third = residue['original+attn+feed'].cpu().detach().numpy()[0][i].mean()
    values.append(attn/first)
    values.append(feed/second)
    # values.append(third)
  ratioResidual.append(values)
    


import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams["figure.figsize"] = [7.50, 5.50]
plt.rcParams["figure.autolayout"] = True
for i in range(len(symbol)):
  y = np.array(ratioResidual[i])
  plt.plot(y, label = symbol[i])
# y2 = np.array([6, 2, 7, 11])
leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(prompt)

plt.savefig(directory + '/ratio_of_residual_Values')
plt.show()