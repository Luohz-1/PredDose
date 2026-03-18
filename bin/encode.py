import numpy as np
from sklearn.preprocessing import OneHotEncoder
import constants

def enc_int_seqs_from_variants(variants, wild_type_seq, wt_offset=0):   # 得到突变后的蛋白质序列矩阵
    # convert wild type seq to integer encoding
    wild_type_int = np.zeros(len(wild_type_seq), dtype=np.uint8)
    for i, c in enumerate(wild_type_seq):
        wild_type_int[i] = constants.C2I_MAPPING[c] # 返回每个氨基酸在 one-hot 编码的位置

    seq_ints = np.tile(wild_type_int, (len(variants), 1))
    # 得到与数据集长度相同的氨基酸矩阵，行（10805）数据集（蛋白个数）长度，列（237）蛋白质长度（氨基酸的个数）
    for i, variant in enumerate(variants):
        # special handling if we want to encode the wild-type seq
        # the seq_ints array is already filled with WT, so all we have to do is just ignore it
        # and it will be properly encoded
        if variant == "_wt":    # 不处理野生型
            continue

        # variants are a list of mutations [mutation1, mutation2, ....]
        variant = variant.split(",")
        for mutation in variant:
            # mutations are in the form <original char><position><replacement char>
            position = int(mutation[1:-1])  # 提取其中氨基酸位置
            replacement = constants.C2I_MAPPING[mutation[-1]]   # 得到突变后的氨基酸在 one-hot 编码中的位置
            seq_ints[i, position-wt_offset] = replacement   # 在氨基酸矩阵 seq_ints 中的氨基酸对应位置修改为突变后的类型

    return seq_ints

def enc_int_seqs_from_char_seqs(char_seqs):
    seq_ints = []
    for char_seq in char_seqs:
        int_seq = [constants.C2I_MAPPING[c] for c in char_seq]
        seq_ints.append(int_seq)
    seq_ints = np.array(seq_ints)
    return seq_ints


def encode_int_seqs(char_seqs=None, variants=None, wild_type_aa=None, wild_type_offset=None):   # 得到突变后的氨基酸矩阵
    single = False
    if variants is not None:
        if not isinstance(variants, list):
            single = True   # 如果变异不是以列表的形式存在，那么将其转化为列表
            variants = [variants]

        int_seqs = enc_int_seqs_from_variants(variants, wild_type_aa, wild_type_offset) # 得到突变后的氨基酸矩阵

    elif char_seqs is not None:
        if not isinstance(char_seqs, list):
            single = True
            char_seqs = [char_seqs]

        int_seqs = enc_int_seqs_from_char_seqs(char_seqs)

    return int_seqs, single

def enc_one_hot(int_seqs):
    enc = OneHotEncoder(categories=[range(constants.NUM_CHARS)] * int_seqs.shape[1], dtype=bool, sparse_output=False)
    one_hot = enc.fit_transform(int_seqs).reshape((int_seqs.shape[0], int_seqs.shape[1], constants.NUM_CHARS))
    return one_hot  # OneHotEncoder 首先初始化，fit_transform 对数据进行转换，结果仍是展平的一维向量，
    # 初始化中的 cateegories 限定了对一个蛋白中多少个氨基酸编码，以及每个氨基酸按照多少种 ONE-HOT 维度编码
    # 如原来为两个蛋白质样本，长度为三，即len=（2,3），使用 fit_transform 之后维度为（2,3*21），所以需要将其转化为（2,3,21）维度

def enc_aa_index(int_seqs): # AA_index 特征
    """ encodes data in aa index properties format """
    aa_features = np.load("../data/pca-19.npy")    # 加载预存的 np 变量，二十个氨基酸对应的19个从114维度的特征降维之后的典型特征
    # add all zero features for stop codon
    aa_features = np.insert(aa_features, 0, np.zeros(aa_features.shape[1]), axis=0) # 插入终止子的对应19维特征向量
    aa_features_enc = aa_features[int_seqs]
    return aa_features_enc

def combine_diploid(list1, list2):  # 将单倍型数据矩阵按照不同的vk基因合并
    combine_list = []
    for i in range(3):
        combine_list.append(np.stack([list1[i], list2[i]], axis=1))
    return combine_list


    # 为ie数据集添加 vkorc1 基因特征
def add_genotype_features(data, encoding_type='one_hot'):
    # 定义基因状态的编码
    genotype_encodings = {
        'one_hot': {
            'GG': np.array([1, 0, 0], dtype=np.float32),
            'GA': np.array([0, 1, 0], dtype=np.float32),
            'AA': np.array([0, 0, 1], dtype=np.float32)
        },
        'binary': {
            'GG': np.array([0, 0], dtype=np.float32),
            'GA': np.array([1, 0], dtype=np.float32),
            'AA': np.array([1, 1], dtype=np.float32)
        },
        'continuous': {
            'GG': np.array([0.3], dtype=np.float32),
            'GA': np.array([0.7], dtype=np.float32),
            'AA': np.array([1.0], dtype=np.float32)
        }
    }

    # 获取当前编码方式的基因状态编码
    encodings = genotype_encodings[encoding_type]

    # 创建一个列表来存储三个数据集
    datasets = []
    vk_data = []
    # 遍历每种基因状态，添加对应的编码特征
    for genotype, encoding in encodings.items():
        # 复制原始数据集
        new_data = np.copy(data)

        # 添加新的特征维度
        new_features = np.tile(encoding, (new_data.shape[0], new_data.shape[1], 1))
        new_data = np.concatenate((new_data, new_features), axis=2) # 将两个维度连接在一起
        vk_features = np.tile(encoding, (new_data.shape[0], 1))

        # 将处理后的数据集添加到列表中
        datasets.append(new_data)
        vk_data.append(vk_features)

    return datasets , vk_data


def encode(encoding, graph_fn, char_seqs=None, variants=None, ds_name=None, wt_aa=None, wt_offset=None):
    """ the main encoding function that will encode the given sequences or variants and return the encoded data """

    if variants is None and char_seqs is None:
        raise ValueError("must provide either variants or full sequences to encode")
    if variants is not None and ((ds_name is None) and ((wt_aa is None) or (wt_offset is None))):
        raise ValueError("if providing variants, must also provide (wt_aa and wt_offset) or "
                         "ds_name so I can look up the WT sequence myself")

    if ds_name is not None: # 如果提供了数据名称，可以直接从 constants 收集蛋白信息
        wt_aa = constants.DATASETS[ds_name]["wt_aa"]
        wt_offset = constants.DATASETS[ds_name]["wt_ofs"]

    # convert given variants or char sequences to integer sequences
    # this may be a bit slower, but easier to program
    int_seqs, single = encode_int_seqs(char_seqs=char_seqs, variants=variants,
                                       wild_type_aa=wt_aa, wild_type_offset=wt_offset)  # 得到突变后的氨基酸矩阵

    # encode variants using int seqs
    encodings = encoding.split(",")
    encoded_data = []
    for enc in encodings:
        if enc == "one_hot":
            encoded_data.append(enc_one_hot(int_seqs))  # 对数据进行 onehot 编码，（蛋白个数，237,23）
        elif enc == "aa_index":
            encoded_data.append(enc_aa_index(int_seqs)) # 添加 AA_index 的19维特征
        else:
            raise ValueError("err: encountered unknown encoding: {}".format(enc))

    # concatenate if we had more than one encoding
    if len(encoded_data) > 1:
        encoded_data = np.concatenate(encoded_data, axis=-1)    # 将 onehot 21维特征与 AA_index 19维特征合并
    else:
        encoded_data = encoded_data[0]

    # if we were passed in a single sequence, remove the extra dimension
    if single:
        encoded_data = encoded_data[0]

    datasets, vk_data = add_genotype_features(encoded_data, encoding_type='one_hot')  # 添加one-hot编码
    datasets_code = np.concatenate((datasets[0], datasets[1], datasets[2]), axis=0)
    datasets_code_vk = np.concatenate((vk_data[0], vk_data[1], vk_data[2]), axis=0)


    datasets = {}
    if not graph_fn == "":
        datasets['encoded_data'] = np.tile(encoded_data, (3, 1, 1))
        datasets['encoded_data'] = np.stack((datasets['encoded_data'], datasets['encoded_data']), axis=1)
        datasets['vk_coding'] = datasets_code_vk
    else:
        datasets['encoded_data'] = datasets_code
        datasets['encoded_data'] = np.stack((datasets['encoded_data'], datasets['encoded_data']), axis=1)   # 得到二倍体数据
        datasets['vk_coding'] = []
    return datasets


def encode_val(encoding, graph_fn, char_seqs=None, variants_1=None, variants_2=None, wt_aa=None, wt_offset=None):

    int_seqs_1, single1 = encode_int_seqs(char_seqs=char_seqs, variants=variants_1,
                                       wild_type_aa=wt_aa, wild_type_offset=wt_offset)  # 得到突变后的氨基酸矩阵
    int_seqs_2, single2 = encode_int_seqs(char_seqs=char_seqs, variants=variants_2,
                                       wild_type_aa=wt_aa, wild_type_offset=wt_offset)  # 得到突变后的氨基酸矩阵

    # encode variants using int seqs
    encodings = encoding.split(",")
    encoded_data_1 = []
    encoded_data_2 = []
    for enc in encodings:
        if enc == "one_hot":
            encoded_data_1.append(enc_one_hot(int_seqs_1))
            encoded_data_2.append(enc_one_hot(int_seqs_2))
        elif enc == "aa_index":
            encoded_data_1.append(enc_aa_index(int_seqs_1)) # 添加 AA_index 的19维特征
            encoded_data_2.append(enc_aa_index(int_seqs_2))  # 添加 AA_index 的19维特征
        else:
            raise ValueError("err: encountered unknown encoding: {}".format(enc))

    # concatenate if we had more than one encoding
    if len(encoded_data_1) > 1:
        encoded_data_1 = np.concatenate(encoded_data_1, axis=-1)    # 将 onehot 21维特征与 AA_index 19维特征合并
        encoded_data_2 = np.concatenate(encoded_data_2, axis=-1)    # 将 onehot 21维特征与 AA_index 19维特征合并
    else:
        encoded_data = encoded_data_1[0]

    # if we were passed in a single sequence, remove the extra dimension
    if single1:
        encoded_data = encoded_data[0]

    datasets_code_1, datasets_code_vk_1 = add_genotype_features(encoded_data_1, encoding_type='one_hot')  # 添加one-hot编码
    datasets_code_2, datasets_code_vk_2 = add_genotype_features(encoded_data_2, encoding_type='one_hot')

    combine = combine_diploid(datasets_code_1, datasets_code_2) # 将单倍型数据矩阵按照不同的vk基因合并

    datasets = {
        'GG_data': [],
        'GA_data': [],
        'AA_data': []
    }
    if not graph_fn == "":
        pass
        # datasets['encoded_data'] = np.tile(encoded_data, (3, 1, 1))
        # datasets['encoded_data'] = np.stack((datasets['encoded_data'], datasets['encoded_data']), axis=1)
        # datasets['vk_coding'] = datasets_code_vk
    else:
        datasets['GG_data'] = combine[0]
        datasets['GA_data'] = combine[1]
        datasets['AA_data'] = combine[2]

    return datasets
