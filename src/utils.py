import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors
import random
from collections import OrderedDict

random.seed(0)

# color_list = list(mcolors.CSS4_COLORS.keys())
# shuffled_color_list = color_list.copy()
# random.shuffle(shuffled_color_list) #this is for the bar plot
shuffled_color_list = [
    "blue", "green", "red", "purple", "orange",
    "brown", "cyan", "magenta", "lime", "navy",
    "teal", "coral", "olive", "maroon", "gold",
    "turquoise", "indigo", "violet", "salmon", "slategray",
    "darkgreen", "chocolate", "crimson", "darkblue", "forestgreen",
    "darkorange", "darkred", "deepskyblue", "firebrick", "midnightblue",
    "orchid", "sienna", "royalblue", "saddlebrown", "tomato",
    "yellowgreen", "steelblue", "rosybrown", "plum", "peru",
    # Hex colors
    "#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", 
    "#f95d6a", "#ff7c43", "#ffa600", "#488f31", "#6c9e49",
    "#91ad61", "#b5bc79", "#d9cb91", "#fdfaa9", "#8b1a1a", 
    "#a52a2a", "#bc3334", "#d24d4d", "#e86666", "#ff7f7f", 
    "#483d8b", "#6a5acd", "#7b68ee", "#9370db", "#b0c4de",
    "#3cb371", "#2e8b57", "#228b22", "#008000", "#006400",
    "#556b2f", "#66cdaa", "#8fbc8f", "#20b2aa", "#008b8b",
    "#00ced1", "#00ffff", "#00ffff", "#00bfff", "#1e90ff",
    "#4682b4", "#5f9ea0", "#6495ed", "#7b68ee", "#7fffd4",
    "#87ceeb", "#87cefa", "#add8e6", "#b0c4de", "#b0e0e6",
    "#f08080", "#e9967a", "#ffa07a", "#ffdead", "#ffdab9",
    "#ffe4b5", "#ffebcd", "#f5deb3", "#deb887", "#d2b48c",
    "#bc8f8f", "#f4a460", "#daa520", "#b8860b", "#cd853f",
    "#d2691e", "#8b4513", "#a0522d", "#a52a2a", "#800000"
]



### 1.analyse_mlm.py -> Compute attribution scores


def get_evaldataset(tmp_data_path, data_path, debug):
    """
    Load the evaluation dataset from tmp_data_path or prepare it from data_path and save it to tmp_data_path.

    returns eval_bag_list_perrel (dict): each key is a relation and teh corresponding a value is a list with
        n head-tail entity pairs, where each of these pairs has m different prompts.
    """
    # prepare eval set
    if os.path.exists(tmp_data_path):
        with open(tmp_data_path, "r") as f:
            eval_bag_list_perrel = json.load(
                f
            )  # json file with a list of diverse prompts for each head-tail entity pair
            # for each relation type in the eval dataset
    else:
        with open(data_path, "r") as f:
            eval_bag_list_all = json.load(f)
        # split bag list into relations
        eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(eval_bag_list_all):
            bag_rel = eval_bag[0][2].split("(")[0]
            if bag_rel not in eval_bag_list_perrel:
                eval_bag_list_perrel[bag_rel] = []
            if len(eval_bag_list_perrel[bag_rel]) >= debug:
                continue
            eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(tmp_data_path, "w") as fw:
            json.dump(eval_bag_list_perrel, fw, indent=2)

    return eval_bag_list_perrel


def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features"""
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example[0])
    # All templates are simple, almost no one will exceed the length limit.
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[: max_seq_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(
        tokens
    )  # all zeros, while the rest only have 0's for padding

    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "baseline_ids": baseline_ids,
    }
    tokens_info = {
        "tokens": tokens,
        "relation": example[2],
        "gold_obj": example[1],
        "pred_obj": None,
    }

    if len(example) == 4: #for file named "all_data_used_bags", which has the predicted label for each prompt example
        tokens_info["pred_obj"] = example[3] #predicted label
        assert example[3] is not None, "Predicted label in the saved eval example is None"

    return features, tokens_info


## 2_get_kn


def pos_list2str(pos_list):
    """
    Transforms a list of positions [layer_id, neuron_id] in a string
    layer_id@neuron_id
    """
    return "@".join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    """
    Transforms a string of positions layer_id@neuron_id in a list
    [layer_id, neuron_id]
    """
    return [int(pos) for pos in pos_str.split("@")]


def stat(data, pos_type, rel):
    if pos_type == "kn_rel":
        print(f"{rel}'s {pos_type} has {len(data)} imp pos. ")
        return
    ave_len = 0
    for kn_bag in data:
        ave_len += len(kn_bag)
    ave_len /= len(data)
    print(f"{rel}'s {pos_type} has on average {ave_len} imp pos. ")


## 2_analyze_kn.py


def analyze_avg_num_kn_perrel(kn_bag_list, prefix=""):
    tot_bag_num, tot_kneurons = 0, 0
    kn_bag_layer_counter = Counter()

    # kn_bag_list = kn positions per entity pair in the relation
    for kn_bag in kn_bag_list:  # for each entity pair
        for kn in kn_bag:  # for each kn pos
            kn_bag_layer_counter.update([kn[0]])  # saving the layer's id
    tot_bag_num += len(kn_bag_list)  # number of entity pairs considered

    for k, v in kn_bag_layer_counter.items():
        tot_kneurons += v
    # tot_kneurons = total no of kn encountered in this relation
    for k, v in kn_bag_layer_counter.items():
        kn_bag_layer_counter[
            k
        ] /= tot_kneurons  # this gives the % of kn that belong to this layer id k

    # average # Kneurons
    avg = tot_kneurons / tot_bag_num
    print(f"average {prefix} ig_kn per entity_pair: ", avg)
    return avg, kn_bag_layer_counter


def plot_kn_dist_over_layers(rel, kn_bag_counter, fig_dir, prefix=""):
    plt.figure(figsize=(8, 5))

    x = np.array([i + 1 for i in range(12)])  # 12 layers in the model
    y = np.array([kn_bag_counter[i] for i in range(12)])  # % of kn per layer id
    plt.xlabel("Layer", fontsize=20)
    plt.ylabel("Percentage", fontsize=20)
    plt.xticks([i for i in range(12)], labels=[i for i in range(12)], fontsize=20)
    plt.yticks(
        np.arange(-0.4, 0.5, 0.1),
        labels=[f"{np.abs(i)}%" for i in range(-40, 50, 10)],
        fontsize=14,
    )
    plt.tick_params(
        axis="y",
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        rotation=0,
        labelsize=14,
    )
    plt.ylim(-y.max() - 0.03, y.max() + 0.03)
    plt.xlim(0.3, 12.7)
    plt.title(f"{prefix}{rel}")
    bottom = -y
    y = y * 2
    plt.bar(x, y, width=1.02, color="#0165fc", bottom=bottom)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)

    plt.savefig(
        os.path.join(fig_dir, f"kneurons_distribution_{rel}_{prefix}.png"), dpi=100
    )


def cal_intersec(kn_bag_1, kn_bag_2):
    kn_bag_1 = set(["@".join(map(str, kn)) for kn in kn_bag_1])
    kn_bag_2 = set(["@".join(map(str, kn)) for kn in kn_bag_2])
    return len(kn_bag_1.intersection(kn_bag_2))


def compute_inner_rel_intersection(kn_pos_perrel, prefix=""):
    inner_avg_intersec = []
    for rel, values in kn_pos_perrel.items():
        print(f"calculating {rel}")
        kn_bag_list = values[f"{prefix}kn_bag_list"]
        len_kn_bag_list = len(kn_bag_list)

        inner_rel_avg_intersec = []
        # computing intersection of kn between different entity pairs of the same relation
        for i in range(0, len_kn_bag_list - 1):
            for j in range(i + 1, len_kn_bag_list):
                kn_bag_1 = kn_bag_list[i]
                kn_bag_2 = kn_bag_list[j]
                num_intersec = cal_intersec(kn_bag_1, kn_bag_2)
                inner_rel_avg_intersec.append(num_intersec)

        mean_inner_rel_avg_intersec = np.array(inner_rel_avg_intersec).mean()
        print(
            f"ig kn for {rel} relation has on average {mean_inner_rel_avg_intersec} inner kn intersection per entity pair."
        )
        inner_avg_intersec.extend(inner_rel_avg_intersec)

    mean_inner_avg_intersec = np.array(inner_avg_intersec).mean()
    print(
        f"ig kn has on average {mean_inner_avg_intersec} inner kn intersection, across all relations"
    )


def plot_kn_pos(kn_bag_list, kn_rel, rel, fig_dir, prefix=""):
    # kn_bag_list = kn_pos_perrel[rel]["kn_bag_list"]
    layers, neurons, layers_rel, neurons_rel = [], [], [], []
    legends = {}

    for i, entity in enumerate(kn_bag_list):
        for layer, pos in entity:
            if (layer, pos) not in legends.keys():
                legends[(layer, pos)] = f"{i}"
                layers.append(layer)
                neurons.append(pos)
            else:
                legends[(layer, pos)] += f";{i}"

    for layer, pos in kn_rel:
        layers_rel.append(layer)
        neurons_rel.append(pos)

    _, b = plt.subplots(1, 2, figsize=(15, 7))
    b[0].scatter(x=layers, y=neurons)
    for k, txt in legends.items():
        b[0].annotate(txt, (k[0] + 0.005, k[1] + 5))
    b[0].set_ylabel("Neuron position")
    b[0].set_xlabel("Layer id")
    b[0].set_title(f"Relation {rel}: {prefix}kn_bag_list (per entity)")
    b[0].set_xticks(np.array(layers, dtype=int))
    b[0].set_yticks(np.array(neurons, dtype=int))

    b[1].scatter(x=layers_rel, y=neurons_rel, marker="D")
    b[1].set_title(f"Relation {rel}: {prefix}kn_rel")
    b[1].set_ylabel("Neuron position")
    b[1].set_xlabel("Layer id")
    b[1].set_xticks(np.array(layers, dtype=int))
    b[1].set_yticks(np.array(neurons, dtype=int))

    os.makedirs(fig_dir, exist_ok=True)

    plt.savefig(
        os.path.join(fig_dir, f"kn_pos_{rel}_{prefix}.png"), dpi=100
    )


def box_plots(kn_bag_list, rel, fig_dir, prefix=''):
    all_neurons, per_layer_neurons = [], {}
    data = []
    for ent in kn_bag_list:
        for (l, n) in ent:
            all_neurons.append(n)
            if l not in per_layer_neurons.keys():
                per_layer_neurons[l] = []
            per_layer_neurons[l].append(n)

    sorted_layers_dict = OrderedDict(sorted(per_layer_neurons.items()))
    labels = ["All"] + [l for l in sorted_layers_dict]
    layers_data = [v for _, v in sorted_layers_dict.items()]
    data.append(all_neurons)
    data.extend(layers_data)

    stats = {}
    for lbl, neurons in zip(labels, data):
        stats[lbl] = {"Unique": 0, "Counts": [], "Mean": 0, "STD": 0, "Range": []}
        array_data = np.asarray(neurons)
        neuron_pos, counts = np.unique(array_data, return_counts=True)
        stats[lbl]["Unique"] = neuron_pos.shape[0]
        stats[lbl]["Counts"] = [counts.min(), counts.max()]
        #stats[lbl]["Min"] = array_data.min()
        #stats[lbl]["Max"] = array_data.max()
        stats[lbl]["Mean"] = array_data.mean()
        stats[lbl]["STD"] = array_data.std()
        stats[lbl]["Range"] = [array_data.mean() - array_data.std(), array_data.mean() + array_data.std()]
        
    plt.figure(figsize=(8, 5))
    plt_dict = plt.boxplot(data, labels=labels)
    plt.title(f"{rel}-{prefix}kn_bag_list")
    plt.xlabel("Layer index")
    plt.ylabel("Neuron position")
    plt.savefig(
        os.path.join(fig_dir, f"kn_bag_list_boxplot_{rel}_{prefix}.png"), dpi=100
    )

    del plt_dict
    
    return stats




def kn_barplot(kn_bag_list, kn_rel, rel, stats, fig_dir, th=12, prefix=""):
    # kn_bag_list = kn_pos_perrel[rel]["kn_bag_list"]
    # th: maximum number of neurons per layer to plot (if surpasses, filter out the less common neurons across entities)
    # legends: neuron postions and assigned bar color
    # layers: keys = layer id, values = {neuron_pos: num_entities/num_occurences}
    legends, layers = {}, {}
    colors = shuffled_color_list

    def create_plt_data_dict(kn_list, layers, legends):
        for layer, neuron in kn_list:
            if layer not in layers.keys():
                layers[layer] = {}
                if neuron not in layers[layer].keys():
                    layers[layer][neuron] = 0
                layers[layer][
                    neuron
                ] += 1  # count no of entities where this (layer, neuron) occur

            else:
                if neuron not in layers[layer].keys():
                    layers[layer][neuron] = 0
                layers[layer][
                    neuron
                ] += 1  # count no of entities where this (layer, neuron) occur

            if neuron not in legends.keys():
                legends[neuron] = 0

        return legends, layers

    def filter_layers(layers, legends, th=12):
        for l, neurons_counts in layers.items():
            if len(neurons_counts.keys()) > th: #th = number of neurons per layer
                print(f" *** Filtering layer {l} ***")
                counts_th = stats[l]["Counts"][-1] * 0.2 #max number of counts * 0.2
                removed_pos = []

                for pos, count in neurons_counts.items():
                    if count <= counts_th: #if that neuron occurs less times than counts_th
                        removed_pos.append(pos)
                for key in removed_pos: 
                    layers[l].pop(key, None) # None is the default value if the key is not found
                    legends.pop(key, None)
                print( f"     --> {len(removed_pos)} neurons removed: {removed_pos}")

                del removed_pos
        return layers, legends

    # kns per entity
    for entity in kn_bag_list:
        legends, layers = create_plt_data_dict(
            entity, layers, legends
        )

    layers, legends = filter_layers(layers, legends, th=th)

    # common kns of the whole relation
    layers_rel = {}
    legends, layers_rel = create_plt_data_dict(
        kn_rel, layers_rel, legends
    )

    # attribute 1 color per neuron position
    color_i = 0
    for neuron_pos in legends.keys():
        legends[neuron_pos] = colors[color_i]
        color_i += 1

    # Function to calculate bar positions
    def calculate_positions(data, bar_width, spacing):
        positions = {}
        offset = 0
        for category, neurons in data.items():
            num_bars = len(neurons.keys())
            total_width = num_bars * bar_width + (num_bars - 1) * spacing
            start_pos = offset - total_width / 2
            bar_positions = [
                start_pos + i * (bar_width + spacing) for i in range(num_bars)
            ]
            positions[category] = bar_positions
            offset += 1
        return positions

    def make_plot(ax_i, layers, legends, bar_width, spacing):
        # Calculate positions for each bar
        positions = calculate_positions(layers, bar_width, spacing)
        # Plotting the bars
        for l, neurons in layers.items():
            print(f"Layer {l}; neurons: {neurons}")
            for i, (n_pos, value) in enumerate(neurons.items()):
                ax[ax_i].bar(
                    positions[l][i],
                    value,
                    bar_width,
                    label=f"N_{n_pos}",
                    color=legends[n_pos],
                )

        # Customizing the axes
        ax[ax_i].set_xticks(range(len(layers.keys())))
        ax[ax_i].set_xticklabels(layers.keys())
        ax[ax_i].set_xlabel("KN positions (layer and neuron)")
        ax[ax_i].set_ylabel("No of entities")
        ax[ax_i].legend(ncol=2)

    # Create figure and axis
    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Initial bar width and spacing
    bar_width = 0.1
    spacing = 0

    for ax_i, data in enumerate([layers, layers_rel]):
        make_plot(ax_i, OrderedDict(sorted(data.items())), legends, bar_width, spacing)

    # Show the plot
    ax[0].set_title(f"Relation {rel}: {prefix}kn_bag_list")
    ax[1].set_title(f"Relation {rel}: {prefix}kn_rel")
    plt.tight_layout()
    plt.show()

    os.makedirs(fig_dir, exist_ok=True)

    plt.savefig(
        os.path.join(fig_dir, f"{prefix}kn_barplot_{rel}.png"), dpi=100
    )


def plot_weights(kn_bag, model):
    layers, neurons = [], {}
    for layer, pos in kn_bag:
        if layer not in layers:
            layers.append(layer)
            neurons[layer] = []
        neurons[layer].append(pos)

    if len(layers) > 1:
        _, ax = plt.subplot(len(layers), figsize=(20, 10))
        for i, l in enumerate(layers):
            w = model.bert.encoder.layer[l].output.dense.weight.detach().numpy()
            ax[i].imshow(w)
            ax[i].set_title(f"Trasnformer Layer {i}")
            ax[i].set_ylabel("Input Tokens")
            ax[i].set_xlabel("FFN hidden size (3072)")
            pos = np.array(neurons[l], dtype=int)
            ax[i].set_xticks(pos)

    else:
        plt.figure(figsize=(20, 10))
        l = layers[0]
        w = model.bert.encoder.layer[l].output.dense.weight.detach().cpu().numpy()
        for pos in neurons[l]:
            print(w[:, pos].min(), w[:, pos].max(), w[:, pos].mean())

        cax = plt.imshow(w, cmap="inferno", interpolation="nearest")
        plt.title(f"Trasnformer Layer {l}")
        plt.ylabel("Model's dimension")
        plt.xlabel("FFN hidden size (3072)")
        pos = np.array(neurons[l], dtype=int)
        plt.xticks(pos)
        plt.colorbar(cax, boundaries=np.linspace(np.min(w), np.max(w), 1000))

    plt.tight_layout()
