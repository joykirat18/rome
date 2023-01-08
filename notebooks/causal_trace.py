
# Commented out IPython magic to ensure Python compatibility.
# %%bash
# !(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit
# cd /content && rm -rf /content/rome
# git clone https://github.com/kmeng01/rome rome > install.log 2>&1
# pip install -r /content/rome/scripts/colab_reqs/rome.txt >> install.log 2>&1
# pip install --upgrade google-cloud-storage >> install.log 2>&1
# prompt = ""
# subject = ""
# kind = "" # mlp or attn 
# noise = 0
# base_directory = ""
# directory = ""
# model_layer_size = 0


import os, re, json
# import config
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
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

torch.set_grad_enabled(False)


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        # print(layer)
        # print(tokens_to_mix)
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                # print(torch.from_numpy(
                #     prng.randn(x.shape[0] - 1, e - b, x.shape[2])))
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp, output_hidden_states=True, output_attentions=True)
    layer_name = []
    for name, param in model.named_parameters():
        layer_name.append(name)

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        for layer in trace_layers:
            # print(untuple(td[layer].output).detach().cpu().shape)
            all_traced = torch.stack(
                [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
            )
        return probs, outputs_exp

    return probs, outputs_exp




def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None, directory = None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score, _ = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    )
    low_score.item()
    if not kind:
        print(kind)
        print(e_range)
        differences, best_model_list = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, kind, noise=noise, directory = directory
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
            directory = directory
        )
    differences = differences.detach().cpu()
    # trace_differences = trace_differences.detach().cpu()
    return dict(
        scores=differences,
        # td = td,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, kind, noise=0.1, directory = None):
    ntoks = inp["input_ids"].shape[1]
    table = []
    trace_table = []
    td_table = []
    
    best_model_list = []
    for tnum in range(ntoks):
        row = []
        trace_row = []
        td_row = []
        best_prob = 0
        best_model = model
        for layer in range(0, num_layers):
            # print([(tnum, layername(model, layer))])
            r, output = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                trace_layers = []
            )

            # print(type(output.state_dict()))
            trace_row.append(output)
            row.append(r)
            if(best_prob < r):
                best_prob = r
                best_model = output
        if(kind is None):
          kind = 'NONE'
        print(tnum)
        print(directory + '/' + kind + '/best_model' + str(tnum) + '.pt')
        torch.save(best_model, directory  + '/' + kind + '/best_model' + str(tnum) + '.pt')
        # torch.save(best_model.state_dict(), str(ntoks))
        # model_scripted = torch.jit.script(best_model)
        # model_scripted.save('best_model' + str(tnum) + '.pt')
        best_model_list.append(best_model)
        table.append(torch.stack(row))
        # td_table.append(torch.stack(td_row))
        # trace_table.append(trace_row)
        
    return torch.stack(table), best_model_list


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1 , directory = None
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        best_prob = 0
        best_model = model
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            # print(layerlist)
            r, output = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
            if(best_prob < r):
                best_prob = r
                best_model = output
        table.append(torch.stack(row))
        if(kind is None):
            kind = 'NONE'
        print(directory + '/' + kind + '/best_model' + str(tnum) + '.pt')
        torch.save(best_model, directory + '/' + kind + '/best_model' + str(tnum) + '.pt')
        print(tnum)
    return torch.stack(table)


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=1,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    for kind in [None, "mlp", "attn"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind
        )

def get_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=5,
    noise=0.1,
    window=5,
    kind=None,
    modelname=None,
    savepdf=None,
    directory = None
):
    if subject is None:
        subject = guess_subject(prompt)
        print(subject)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind, directory = directory
    )
    # print(result)
    filename = prompt + "_" + str(noise) + ".npz"
    numpy_result = {
    k: v.detach().cpu().numpy() if torch.is_tensor(v) and k != 'best_model_list' else v
    for k, v in result.items()
    }
    # numpy.savez(base_directory + '/' + filename, **numpy_result)
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def get_all_results(mt, prompt, kind, subject=None, noise=0.1, modelname=None, directory = None):
    # for kind in [None, "mlp", "attn"]:
      get_hidden_flow(
          mt, prompt, subject, modelname=modelname, noise=noise, kind=kind, savepdf = directory + '/result', directory = directory
      )

def model_hidden_state(mt, prompt):
    inp = make_inputs(mt.tokenizer, prompt)
    outputs_exp = mt.model(**inp, output_hidden_states=True, output_attentions=True)
    return outputs_exp



def get_model(directory):
    import torch
    model = torch.load(directory)
    return model

def get_mean_hidden_state(model, hidden_number, token_number):
    return model[2][hidden_number].mean(dim=0).cpu().numpy()[token_number]

def compare_hidden_states(a, b):
    if(a.shape != b.shape):
        print("hidden state size not equal")
        return NULL
    diff = b - a
    return diff
def Average(lst):
    return sum(lst) / len(lst)

def plot_array(array,prompt,vmin, vmax, title, directory, kind):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    np.random.seed(0)
    sns.set()
    # uniform_data = np.random.rand(10, 12)
    ax = sns.heatmap(array, vmin = vmin, vmax = vmax)
    plt.xlabel('hidden States')
    plt.ylabel(prompt)
    plt.title(title)
    if(kind is None):
        plt.savefig(directory + '/NONE' + '/' + title)
    else:
        plt.savefig(directory + '/' + kind + '/' + title)
    # plt.show()

def get_Min_Max(prompt, directory, kind, model_layer_size):
  vmin = 2147483648
  vmax = -2147483648
  for k in range(len(prompt.split())):
    base_model_directory = directory + '/' + 'base_model.pt'
    if(kind is None):
        model_directory = directory + '/NONE' + '/best_model' + str(k) + '.pt'
    else:
        model_directory = directory + '/' + kind + '/best_model' + str(k) + '.pt'
    hidden_diff = []
    
    base_model = get_model(base_model_directory)
    model = get_model(model_directory)
    min_diff = 2147483648
    max_diff = -2147483648
    for j in range(len(prompt.split())):
        temp = []
        for i in range(model_layer_size):
            base_state = get_mean_hidden_state(base_model, i,j)
            state = get_mean_hidden_state(model, i, j)
            diff = compare_hidden_states(base_state, state)
            mean_diff = Average(diff)
            # print(mean_diff)
            temp.append(mean_diff)
            min_diff = min(min_diff, mean_diff)
            max_diff = max(max_diff, mean_diff)
        hidden_diff.append(temp)
    import math
    # print(math.floor(min_diff), math.ceil(max_diff))
    vmin = min(math.floor(min_diff), vmin)
    vmax = max(math.ceil(max_diff), vmax)
  # print(vmin, vmax)
  return vmin, vmax

# get_Min_Max(prompt)

def plot_diff_array(prompt, directory, kind, model_layer_size, token_changed, vmin, vmax):
    base_model_directory = directory + '/' + 'base_model.pt'
    if(kind is None):
        model_directory = directory + '/NONE' + '/best_model' + str(token_changed) + '.pt'
    else:
        model_directory = directory + '/' + kind + '/best_model' + str(token_changed) + '.pt'
    hidden_diff = []
    
    base_model = get_model(base_model_directory)
    model = get_model(model_directory)
    for j in range(len(prompt.split())):
        temp = []
        for i in range(model_layer_size):
            base_state = get_mean_hidden_state(base_model, i,j)
            state = get_mean_hidden_state(model, i, j)
            diff = compare_hidden_states(base_state, state)
            mean_diff = Average(diff)
            # print(mean_diff)
            temp.append(mean_diff)
        hidden_diff.append(temp)
    import math
    plot_array(hidden_diff,prompt, vmin, vmax,
               "Comparision of original model and model with " + str(token_changed) +  "th token restored_" + str(vmin) + '_' + str(vmax), directory, kind)
    # return hidden_diff

# def diff_list(a, b):
#     diff = []
#     for i in range(len(a)):
#         temp = []
#         for j in range(len(a[0])):
#             temp.append(b[i][j] - a[i][j])
#         diff.append(temp)
#     plot_array(diff, prompt, "")
#     return diff




def main(model_name, prompt, subject, kind, noise, model_layer_size):
    # model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b" or "gpt2-xl" or "EleutherAI/gpt-neo-1.3B"
    mt = ModelAndTokenizer(
        model_name,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
    )

    num_layers = 0
    for name, layer in mt.model.named_modules():
        print(name)
    print(name)

    # prompt = '8 + 8 ='
    # subject = "+"
    # kind = 'attn' # mlp or attn 
    # noise = 0.1
    # model_layer_size = 49

    print(predict_token(mt, [prompt],return_p=True))
    trace_layers = []
    for i in range(0,48):
        trace_layers.append('transformer.h.' + str(i))

    from datetime import datetime

    base_directory = str(datetime.now())
    print(base_directory)
    isExist = os.path.exists(base_directory)
    if not isExist:
        if(kind is None):
            os.makedirs(base_directory+ '/NONE') 
        else:
            os.makedirs(base_directory+ '/' + kind) 
    base_model = model_hidden_state(mt, [prompt])
    # base_model(
    torch.save(base_model, base_directory + "/base_model.pt")

    # base_model[2][1].shape

    print(base_directory)

    weight = base_model[3][1].mean(dim = 0).mean(dim = 0)

    """The following prompt can be changed to any factual statement to trace."""

    get_all_results(mt, prompt,kind, subject = subject, noise=noise, directory = base_directory)

    """Here we trace a few more factual statements from a file of test cases."""

    if(kind is None):
        directory = base_directory
        d = {'prompt':prompt,'subject':subject,'noise':noise}
        with open(directory + '/NONE' +'/' + 'detail.txt', 'w') as f:
            f.write('dict = ' + str(d) + '\n')
    else:
        directory = base_directory
        d = {'prompt':prompt,'subject':subject,'noise':noise}
        with open(directory + '/' + kind +'/' + 'detail.txt', 'w') as f:
            f.write('dict = ' + str(d) + '\n')

    vmin, vmax = get_Min_Max(prompt, directory, kind, model_layer_size)

    for i in range(len(prompt.split())):
        plot_diff_array(prompt,directory, kind, model_layer_size, i, vmin, vmax)
        plot_diff_array(prompt,directory, kind, model_layer_size, i, 0, vmax)
        if(kind is None):
            model_directory = directory + '/NONE' + '/best_model' + str(i) + '.pt'
        else:
            model_directory = directory + '/' + kind + '/best_model' + str(i) + '.pt'
        os.unlink(model_directory)

    base_model_directory = directory + '/' + 'base_model.pt'
    os.unlink(base_model_directory)

if __name__ == '__main__':
    model_name = "gpt2-xl"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b" or "gpt2-xl" or "EleutherAI/gpt-neo-1.3B"
    prompt = '8 + 8 ='
    subject = "+"
    kind = None # None or 'mlp' or 'attn' 
    noise = 0.1
    model_layer_size = 49

    import argparse
 
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-model_name", "--model_name")
    parser.add_argument("-prompt", "--prompt")
    parser.add_argument("-subject", "--subject")
    # parser.add_argument("-kind", "--kind")
    # parser.add_argument("-noise", "--noise")
    parser.add_argument("-model_layer_size", "--model_layer_size")
    
    # Read arguments from command line
    args = parser.parse_args()

    prompt = args.prompt
    model_name = args.model_name
    subject = args.subject
    # kind = args.kind
    # noise = args.noise
    model_layer_size = int(args.model_layer_size)
    print("Prompt " + prompt)
    print("Model Name " + model_name)

    main(model_name, prompt, subject, kind, noise, model_layer_size)

