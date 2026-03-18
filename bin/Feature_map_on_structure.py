"""
因为是双倍型，所以最后输出双倍型的归因值，可以画两个图
"""
import torch
import utils
import encode as enc
import constants
import numpy as np
from copy import deepcopy
from captum.attr import IntegratedGradients
from Bio.PDB import PDBParser, PDBIO


def compute_ig(model, baseline, data, n_steps=100):
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs=data,
        baselines=baseline,
        n_steps=n_steps,
        target=None
    )
    return attributions.sum(dim=-1).squeeze(0)  # Shape: (2, 490)


def aggregate_attributions(model, baseline, data):
    model.to(data.device)
    all_attr = []

    for i in range(data.shape[0]):
        attr = compute_ig(model, baseline, data[i].unsqueeze(0))
        all_attr.append(attr)

    return torch.mean(torch.stack(all_attr), dim=0).cpu().numpy()


def map_to_pdb(pdb_file, attributions, output_prefix='data/cyp2c9/AF'):
    parser = PDBParser()
    structure = parser.get_structure("cyp2c9", pdb_file)

    # 创建三个独立副本
    structures = [deepcopy(structure) for _ in range(3)]

    # 分别映射分支1、分支2和合并归因
    for idx, struct in enumerate(structures[:2]):  # 前两个结构用于分支
        for i, residue in enumerate(struct.get_residues()):
            if i < attributions.shape[1]:
                for atom in residue:
                    atom.set_bfactor(attributions[idx, i] * 100)

    # 第三个结构用于合并归因
    merged_attr = attributions.sum(axis=0)
    for i, residue in enumerate(structures[2].get_residues()):
        if i < len(merged_attr):
            for atom in residue:
                atom.set_bfactor(merged_attr[i] * 100)

    # merged_attr.sort()

    # 获取归因值前十名与后十名氨基酸位点，
    # 前十名：237、332、98、149、328、 133、266、327、167、146（实际氨基酸位置）
    # 后十名：163、72、48、101、337、  93、70、326、346、370
    sorted_indices = np.argsort(-merged_attr)  # 降序排列的索引
    sorted_values = merged_attr[sorted_indices]  # 按索引获取排序后的值

    # 保存文件
    io = PDBIO()
    for i, struct in enumerate(structures):
        io.set_structure(struct)
        io.save(f"{output_prefix}_{i + 1}.pdb")



# 数据加载与编码
ds = utils.load_dataset(ds_fn='data/cyp2c9/cyp2c9_dg.tsv')
wt_aa = constants.DATASETS["cyp2c9"]["wt_aa"]
wt_ofs = constants.DATASETS["cyp2c9"]["wt_ofs"]

data = {
    "scores": np.concatenate([ds[k].to_numpy(dtype='float32') for k in ['score_GG', 'score_GA', 'score_AA']]),
    "variants": ds['variant'].tolist(),
    "encoded_data": enc.encode(
        encoding='one_hot,aa_index',
        graph_fn="",
        variants=ds['variant'].tolist(),
        wt_aa=wt_aa,
        wt_offset=wt_ofs
    )
}

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoded_data = torch.tensor(data["encoded_data"]["encoded_data"], dtype=torch.float32).to(device)

# 模型加载
model = torch.load('output/cnn_3X128_full_model.pth', weights_only=False) # cnn_3X128_full_model.pth
model.eval()

# 计算归因
baseline = torch.zeros(1, 2, 490, 43, device=device)    # 使用零变量矩阵作为基线数据
baseline_wt = enc.encode(
        encoding='one_hot,aa_index',
        graph_fn="",
        variants=['_wt'],
        wt_aa=wt_aa,
        wt_offset=wt_ofs
    )    # 使用野生型数据
baseline_wt = torch.tensor(baseline_wt["encoded_data"], dtype=torch.float32).to(device) # （3,2,490,43）包含三种vk的基因型
mean_attributions = aggregate_attributions(model, baseline, encoded_data)

# 映射到PDB
map_to_pdb(
    pdb_file='data/cyp2c9/AF-P11712-F1-model_v4.pdb',
    attributions=mean_attributions,
    output_prefix='data/cyp2c9/AF'
)
print(1)
