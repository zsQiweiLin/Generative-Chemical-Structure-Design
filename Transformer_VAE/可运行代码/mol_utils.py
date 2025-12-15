# 提供Transformer-VAE分子生成任务的全流程工具函数，包含环境配置、设备检测、分子属性计算、数据预处理
# 自定义Tokenizer/数据集/模型、分子生成及多维度质量评估等核心模块，适配Jupyter Notebook单线程运行环境
import os
import pickle
import warnings
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from multiprocessing import Value, cpu_count
# Transformers 核心库：用于构建Encoder-Decoder架构的Transformer模型
from transformers import (
    PreTrainedTokenizer, PreTrainedModel,
    EncoderDecoderConfig, BertConfig, BertModel
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.tokenization_utils_base import BatchEncoding
# 分子处理库：用于SMILES/SELFIES编码转换、化学属性计算、分子结构可视化
import selfies as sf
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Chem import Draw
import pandas as pd
# Jupyter展示依赖：用于在Notebook中可视化分子结构图
from IPython.display import display
from PIL import Image
import io

# ====================== 基础环境配置 ======================
#   屏蔽各类冗余日志和警告，强制单线程运行，避免Jupyter环境卡死）
# 屏蔽RDKit的所有冗余日志
RDLogger.DisableLog('rdApp.*')
# 屏蔽Python的三类核心警告（UserWarning/版本兼容/数值计算）
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 强制RDKit进入静默模式，关闭OpenMP/MKL多线程避免资源竞争
os.environ["RDKit_SILENT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
# 多线程开关（强制禁用，Notebook环境下多线程易导致进程卡死）
ENABLE_MULTITHREADING = False
MIN_CPU_CORES_FOR_MULTI = 4      # 启用多线程的最小CPU核心数
MIN_MEM_GB_FOR_MULTI = 16        # 启用多线程的最小内存（GB）
MIN_GPU_MEM_GB_FOR_MULTI = 8     # 启用多线程的最小GPU显存（GB）
MAX_SAFE_THREADS = cpu_count() // 2  # 安全线程数上限（CPU核心数的一半）

# ====================== 设备性能检测 ======================
#   检测CPU/GPU/内存等硬件配置，返回性能等级和推荐线程数，为训练配置提供依据
def detect_device_performance():
    """
    检测当前设备的硬件性能，返回性能评估结果
    Returns:
        dict: 包含CPU核心数、内存大小、GPU可用性、显存大小、性能等级、推荐线程数的字典
    """
    cpu_cores = cpu_count()     # 获取CPU核心总数
    # 计算总内存和可用内存（单位转换为GB）
    mem_total = psutil.virtual_memory().total / 1024 ** 3
    mem_available = psutil.virtual_memory().available / 1024 ** 3
    gpu_available = torch.cuda.is_available()   # 判断是否有可用GPU
    gpu_mem = 0.0
    if gpu_available:
        # 获取GPU总显存（单位转换为GB）
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

    # 判定设备性能等级
    performance_level = "high"
    if cpu_cores < MIN_CPU_CORES_FOR_MULTI or mem_available < MIN_MEM_GB_FOR_MULTI:
        performance_level = "low"
    elif gpu_available and gpu_mem < MIN_GPU_MEM_GB_FOR_MULTI:
        performance_level = "medium"
    recommended_workers = 0    # Notebook环境强制单线程，禁用多进程加载
    return {
        "cpu_cores": cpu_cores,
        "mem_total_gb": round(mem_total, 1),
        "mem_available_gb": round(mem_available, 1),
        "gpu_available": gpu_available,
        "gpu_mem_gb": round(gpu_mem, 1) if gpu_available else 0.0,
        "performance_level": performance_level,
        "recommended_workers": recommended_workers
    }


# ====================== 分子核心属性计算 ======================
#   实现分子合成可及性、化学键约束、SMILES合理性过滤、化学规则标签的计算工具
def calculate_SA_score(mol):
    """
    计算分子合成可及性评分（SA），彻底适配低版本RDKit，不依赖官方SA函数
    Args:
        mol: RDKit分子对象
    Returns:
        float: 合成可及性评分（范围0-10，分数越低合成难度越低）
    """
    if mol is None:
        return 10.0
    num_atoms = mol.GetNumAtoms()          # 分子总原子数
    num_bonds = mol.GetNumBonds()          # 分子总化学键数
    num_rings = len(Chem.GetSymmSSSR(mol)) # 分子中环的数量
    # 杂原子数量（排除C/H/O/S等常见原子）
    num_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [6, 1, 8, 16])

    # 各维度评分计算（限制最大值避免单维度主导）
    atom_score = min(num_atoms / 50 * 2.0, 2.0)    # 原子数评分
    ring_score = min(num_rings / 5 * 3.0, 3.0)     # 环数评分
    hetero_score = min(num_hetero / max(num_atoms, 1) * 3.0, 3.0)  # 杂原子占比评分
    bond_score = min(num_bonds / max(num_atoms, 1) * 2.0, 2.0)     # 键数/原子数比评分
    total_score = atom_score + ring_score + hetero_score + bond_score
    return min(total_score, 10.0)   # 总分限制在10以内

def set_universal_bond_constraints():
    """设置自定义化学键语义约束，限制超价原子（如S最多4个键、P最多5个键），避免生成化学无效分子"""
    custom_constraints = sf.get_semantic_constraints()   # 获取SELFIES默认约束
    # 自定义超价原子键数约束（适配类药分子常见原子）
    supervalent_constraints = {
        "S": 4, "P": 5, "Cl": 3, "Br": 5, "I": 6,
        "Se": 4, "Te": 6, "As": 5, "Si": 6, "B": 4
    }
    custom_constraints.update(supervalent_constraints)
    sf.set_semantic_constraints(custom_constraints)  # 应用自定义约束


def filter_unreasonable_smiles(smiles):
    """
    过滤化学不合理的SMILES，仅排除剧毒/罕见原子，保留药物常见原子
    Args:
        smiles: 分子的SMILES字符串
    Returns:
        bool: 合理返回True，不合理返回False
    """
    # 基础格式校验：必须是字符串且长度在2-100之间
    if not isinstance(smiles, str) or len(smiles) < 2 or len(smiles) > 100:
        return False
    # 排除含剧毒/罕见原子的分子（Hg/Pb/As等）
    forbidden_atoms = ['Hg', 'Pb', 'As', 'Se', 'Te']
    for atom in forbidden_atoms:
        if atom in smiles:
            return False
    # 化学结构合法性校验（RDKit无法解析则判定为无效）
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    # 价键合理性校验（限制常见原子的最大显式价键）
    valence_constraints = {'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 4, 'Cl': 1, 'F': 1, 'Br': 1, 'I': 1}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in valence_constraints and atom.GetExplicitValence() > valence_constraints[symbol]:
            return False
    # 环大小合理性校验（环原子数需在2-10之间）
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if len(ring) < 2 or len(ring) > 10:
            return False
    # 总电荷合理性校验（绝对值不超过2）
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if abs(total_charge) > 2:
        return False
    return True


def calculate_chemical_rules_label(mol):
    """
    计算分子的化学合理性标签和归一化类药属性，用于模型多任务损失计算
    Args:
        mol: RDKit分子对象
    Returns:
        dict: 包含价键合法性、环合法性、归一化分子量/LogP/QED/SA的标签字典
    """
    # 价键合法性标签（1为合法，0为非法）
    valence_ok = 1
    valence_constraints = {'C': 4, 'H': 1, 'O': 2, 'N': 3, 'S': 4, 'Cl': 1, 'F': 1, 'Br': 1}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in valence_constraints and atom.GetExplicitValence() > valence_constraints[symbol]:
            valence_ok = 0
            break
    # 环合法性标签（环原子数3-8为合法，1为合法，0为非法）
    ring_ok = 1
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if len(ring) < 3 or len(ring) > 8:
            ring_ok = 0
            break
    # 计算类药属性并归一化到合理范围
    mw = Descriptors.MolWt(mol)      # 分子量
    logp = Descriptors.MolLogP(mol)  # 疏水性LogP
    qed = QED.qed(mol)               # QED类药性评分
    sa = calculate_SA_score(mol)     # 合成可及性评分
    return {
        'valence_ok': valence_ok,
        'ring_ok': ring_ok,
        'mw': (mw - 200) / 300,      # 分子量归一化（适配200-500范围）
        'logp': (logp + 2) / 7,      # LogP归一化（适配-2到5范围）
        'qed': qed,                  # QED评分无需归一化（0-1）
        'sa': (sa - 5) / 5           # SA评分归一化（适配0-10范围）
    }


# ====================== 预处理工具（无锁防卡死，全量数据扩容缓存） ======================
#   实现批量SMILES数据预处理，缓存分子属性避免重复计算，适配全量100万条数据规模

# 缓存200万条分子属性，全量数据适配，避免重复计算
@lru_cache(maxsize=2_000_000)  # 小样本为500_000，全量扩容至200万
def calculate_chemical_properties_cached(smiles, TOTAL_DATA_SIZE):
    """
    缓存分子的核心化学属性，跳过耗时的能量计算，提升预处理效率
    Args:
        smiles: 分子SMILES字符串
        TOTAL_DATA_SIZE: 数据集总规模（仅作为缓存键，无实际计算作用）
    Returns:
        tuple/None: 有效分子返回(smiles, selfies, 属性列表)，无效返回None
    """
    mol = Chem.MolFromSmiles(smiles)  # 解析SMILES为分子对象
    if not mol:
        return None
    try:
        # 计算核心类药属性
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        sa_score = calculate_SA_score(mol)
        energy = 300                # 固定能量值，跳过耗时的分子力场计算
    except Exception as e:
        return None
    try:
        selfies = sf.encoder(smiles, strict=False)  # SMILES转SELFIES编码
        return (smiles, selfies, [mw, logp, h_donors, h_acceptors, sa_score, energy]) if selfies else None
    except:
        return None


def batch_process_smiles_large_scale(smiles_list, batch_size=50_000, perf_info=None, TOTAL_DATA_SIZE=1_000_000):  # 小样本为15000，全量提至50000
    """
    批量预处理大规模SMILES数据集，输出带SELFIES和属性的结构化数据，适配全量100万条数据
    Args:
        smiles_list: 原始SMILES列表
        batch_size: 每个预处理批次的样本数（全量适配5万）
        perf_info: 设备性能信息字典
        TOTAL_DATA_SIZE: 数据集总规模
    Returns:
        list: 预处理后的分子数据列表，每个元素为含smiles/selfies/properties的字典
    """
    final_workers = 0  # 强制单线程处理，避免Notebook卡死
    print(f"使用 {final_workers} 线程处理 {len(smiles_list)} 条数据...")
    # 计算总批次数量
    total_batches = (len(smiles_list) + batch_size - 1) // batch_size
    all_processed_data = []     # 存储所有预处理后的有效数据

    # 存储所有预处理后的有效数据
    for batch_idx in tqdm(range(total_batches), desc="预处理批次进度", dynamic_ncols=True):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(smiles_list))
        batch_smiles = smiles_list[start_idx:end_idx]
        print(f"\n处理批次{batch_idx + 1}/{total_batches}，共{len(batch_smiles)}条")
        batch_results = []
        for idx, smi in enumerate(batch_smiles):
            if idx % 500 == 0:      # 每500条输出一次进度，避免日志刷屏
                print(f"  已处理{idx}/{len(batch_smiles)}条")
            res = calculate_chemical_properties_cached(smi, TOTAL_DATA_SIZE)
            batch_results.append(res)

        # 统计当前批次有效分子数量
        valid_in_batch = sum(1 for res in batch_results if res is not None)
        print(f"批次{batch_idx + 1}完成，有效分子{valid_in_batch}/{len(batch_smiles)}")

        # 整理有效数据到结果列表
        for res in batch_results:
            if res is not None:
                all_processed_data.append({"smiles": res[0], "selfies": res[1], "properties": res[2]})
        del batch_smiles, batch_results
    print(f"预处理完成：{len(all_processed_data)}/{len(smiles_list)} 有效")
    return all_processed_data


# ====================== SELFIES Tokenizer ======================
#   自定义适配SELFIES编码的Tokenizer，继承自HuggingFace PreTrainedTokenizer
class SELFIESTokenizer(PreTrainedTokenizer):
    """自定义SELFIES编码的Tokenizer，适配Transformer模型输入格式，支持编码/解码/词汇表保存"""
    vocab_files_names = {"vocab_file": "selfies_vocab.txt"} # 词汇表文件名
    model_input_names = ["input_ids", "attention_mask"]     # 模型输入名称

    def __init__(self, vocab=None, pad_token="<PAD>", sos_token="<SOS>", eos_token="<EOS>", **kwargs):
        """
        初始化SELFIES Tokenizer
        Args:
            vocab: 自定义词汇表字典，None则使用默认初始词汇表
            pad_token: 填充token
            sos_token: 序列起始token
            eos_token: 序列结束token
            **kwargs: 父类初始化参数
        """
        self.sos_token = sos_token  # 序列起始标记
        self.eos_token = eos_token  # 序列结束标记
        self.pad_token = pad_token  # 序列填充标记
        self.unk_token = "<UNK>"    # 未知token

        # 初始化词汇表，默认包含4个特殊token
        if vocab is None:
            self.vocab = {pad_token: 0, sos_token: 1, eos_token: 2, "<UNK>": 3}
        else:
            self.vocab = dict(vocab)

        # 构建ID到token的反向映射
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        # 记录特殊token的ID
        self.pad_id = self.vocab[pad_token]
        self.sos_id = self.vocab[sos_token]
        self.eos_id = self.vocab[eos_token]
        self.unk_id = self.vocab["<UNK>"]
        super().__init__(
            pad_token=pad_token, bos_token=sos_token, eos_token=eos_token, unk_token="<UNK>", **kwargs
        )

    def get_vocab(self):
        """返回词汇表的副本，避免外部修改影响内部状态"""
        return self.vocab.copy()

    @property
    def vocab_size(self):
        """返回词汇表大小"""
        return len(self.vocab)

    def _tokenize(self, text):
        """
        核心分词方法：将SELFIES字符串拆分为单个token
        Args:
            text: SELFIES字符串
        Returns:
            list: 拆分后的token列表
        """
        if not text or text.strip() == "":              # 空字符串返回空列表
            return []
        return list(sf.split_selfies(text))             # 使用SELFIES库拆分token

    def _convert_tokens_to_ids(self, tokens):
        """将token列表转换为ID列表，未知token映射为UNK_ID"""
        return [self.vocab.get(token, self.unk_id) for token in tokens]

    def _convert_ids_to_tokens(self, ids):
        """将token列表转换为ID列表，未知token映射为UNK_ID"""
        if not isinstance(ids, (list, tuple)):
            ids = [ids]
        return [self.id_to_vocab.get(id, "<UNK>") for id in ids]

    def encode(self, selfies, max_length=100, padding="max_length", truncation=True, return_tensors=None):
        """
        将SELFIES字符串编码为模型输入格式（input_ids + attention_mask）
        Args:
            selfies: SELFIES字符串
            max_length: 序列最大长度
            padding: 填充策略（默认max_length）
            truncation: 是否截断过长序列
            return_tensors: 返回张量类型（None为列表，pt为PyTorch张量）
        Returns:
            BatchEncoding: 包含input_ids和attention_mask的编码结果
        """
        tokens = self._tokenize(selfies)  # 分词得到token列表
        # 拼接SOS和EOS标记，构建完整序列
        ids = [self.sos_id] + self._convert_tokens_to_ids(tokens) + [self.eos_id]

        # 截断过长序列
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        # 填充过短序列到max_length
        if padding == "max_length" and len(ids) < max_length:
            ids += [self.pad_id] * (max_length - len(ids))
        # 生成注意力掩码（1为有效token，0为PAD）
        attention_mask = [1 if id != self.pad_id else 0 for id in ids]

        output_dict = {"input_ids": ids, "attention_mask": attention_mask}
        # 转换为PyTorch张量（如需）
        if return_tensors == "pt":
            output_dict["input_ids"] = torch.tensor(output_dict["input_ids"], dtype=torch.long)
            output_dict["attention_mask"] = torch.tensor(output_dict["attention_mask"], dtype=torch.long)
        return BatchEncoding(output_dict)

    def decode(self, ids, skip_special_tokens=True):
        """
        将ID序列解码为SELFIES字符串
        Args:
            ids: ID列表或PyTorch张量
            skip_special_tokens: 是否跳过特殊token（PAD/SOS/EOS/UNK）
        Returns:
            str: 解码后的SELFIES字符串
        """
        # 处理PyTorch张量输入，转为列表
        if isinstance(ids, torch.Tensor):
            if ids.dim() > 1:
                ids = ids.squeeze(0)  # 移除batch维度
            ids = ids.cpu().numpy().tolist()
        if not isinstance(ids, (list, tuple)):
            ids = [ids]

        tokens = self._convert_ids_to_tokens(ids)  # ID转token
        # 跳过特殊token（如需）
        if skip_special_tokens:
            special_tokens = {self.pad_token, self.sos_token, self.eos_token, self.unk_token}
            tokens = [t for t in tokens if t not in special_tokens]
        return "".join(tokens)  # 拼接为完整SELFIES字符串

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
                保存词汇表到指定目录
                Args:
                    save_directory: 保存目录路径
                    filename_prefix: 文件名前缀
                Returns:
                    tuple: 词汇表文件路径元组
                """
        os.makedirs(save_directory, exist_ok=True)  # 创建目录（不存在则新建）
        # 构建词汇表文件名
        vocab_file = os.path.join(
            save_directory, f"{filename_prefix}_" if filename_prefix else "" + self.vocab_files_names["vocab_file"]
        )
        # 按ID升序保存词汇表
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
        return (vocab_file,)


# ====================== 自定义VAE数据集 ======================
#   构建适配Transformer-VAE的数据集，同时预处理化学规则标签，支持模型多任务训练
class SELFIESVAEDataset(Dataset):
    """自定义数据集类，输入SELFIES和SMILES列表，输出模型训练所需的token序列和化学标签"""
    def __init__(self, selfies_list, smiles_list, tokenizer, max_length=100):
        """
        初始化数据集
        Args:
            selfies_list: SELFIES序列列表
            smiles_list: 对应的SMILES序列列表
            tokenizer: SELFIESTokenizer实例
            max_length: SELFIES序列最大长度
        """
        self.selfies_list = selfies_list
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chemical_labels = []  # 存储化学规则标签
        self.valid_idx = []        # 存储有效样本的索引

        # 预处理化学规则标签，显示进度
        for idx, (selfies, smiles) in enumerate(tqdm(zip(selfies_list, smiles_list), desc="预处理化学规则标签")):
            mol = Chem.MolFromSmiles(smiles) # 解析SMILES为分子对象
            if not mol or not selfies:       # 跳过无效样本
                continue
            label = calculate_chemical_rules_label(mol)  # 计算化学标签
            self.chemical_labels.append(label)
            self.valid_idx.append(idx)

        # 过滤无效样本，只保留有效数据
        self.selfies_list = [selfies_list[i] for i in self.valid_idx]
        self.smiles_list = [smiles_list[i] for i in self.valid_idx]
        # 校验样本数与标签数一致
        assert len(self.selfies_list) == len(self.chemical_labels), "样本与标签数不匹配"

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.selfies_list)

    def __getitem__(self, idx):
        """
        获取单个样本数据
        Args:
            idx: 样本索引
        Returns:
            dict: 包含input_ids/attention_mask/labels/化学标签的样本字典
        """
        selfies = self.selfies_list[idx]
        label = self.chemical_labels[idx]
        # 编码SELFIES为模型输入格式
        encoding = self.tokenizer.encode(
            selfies, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].flatten()             # 展平为一维张量
        attention_mask = encoding["attention_mask"].flatten()   # 展平注意力掩码
        labels = input_ids.clone()                              # 标签与输入ID一致（自回归训练）

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "valence_ok": torch.tensor(label['valence_ok'], dtype=torch.float32),
            "ring_ok": torch.tensor(label['ring_ok'], dtype=torch.float32),
            "mw": torch.tensor(label['mw'], dtype=torch.float32),
            "logp": torch.tensor(label['logp'], dtype=torch.float32),
            "qed": torch.tensor(label['qed'], dtype=torch.float32),
            "sa": torch.tensor(label['sa'], dtype=torch.float32)
        }


# ====================== Transformer-VAE模型（全量适配，提升隐层维度） ======================
#   实现融合化学属性的Transformer-VAE模型，支持多任务训练和类药分子生成
class TransformerVAE(PreTrainedModel):
    """融合化学属性的Transformer-VAE模型，支持重构损失、KL损失、化学分类/回归多任务训练，适配类药分子生成"""
    config_class = EncoderDecoderConfig
    def __init__(self, config, latent_dim=128):  # 小样本为64，全量提升至128
        """
        初始化Transformer-VAE模型
        Args:
            config: EncoderDecoderConfig配置实例
            latent_dim: VAE隐空间维度（全量适配128）
        """
        super().__init__(config)
        self.config = config
        self.latent_dim = latent_dim
        # 获取词汇表大小（兼容配置缺失情况）
        self.vocab_size = config.encoder.vocab_size if (
                    hasattr(config.encoder, "vocab_size") and config.encoder.vocab_size > 0) else 1000

        # 初始化Encoder（Bert模型，禁用池化层
        self.encoder_model = BertModel(config.encoder, add_pooling_layer=False)
        # 初始化Decoder（Bert模型，启用解码器模式和交叉注意力）
        decoder_config = config.decoder
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.decoder_model = BertModel(decoder_config, add_pooling_layer=False)

        # 化学属性嵌入层（将4维化学属性映射为32维嵌入）
        self.chem_emb_dim = 32
        self.chem_emb_layer = nn.Linear(4, self.chem_emb_dim)
        # 化学属性融合注意力层（融合Encoder输出和化学嵌入
        self.chem_attention = nn.MultiheadAttention(
            embed_dim=config.encoder.hidden_size + self.chem_emb_dim,
            num_heads=2,
            batch_first=True
        )

        # VAE隐层映射层（均值和方差）
        self.fc_mu = nn.Linear(config.encoder.hidden_size + self.chem_emb_dim, latent_dim)
        self.fc_var = nn.Linear(config.encoder.hidden_size + self.chem_emb_dim, latent_dim)
        # 隐层向量到Decoder输入的映射层
        self.fc_z = nn.Linear(latent_dim, config.decoder.hidden_size)
        # 语言建模头（Decoder输出到词汇表概率）
        self.lm_head = nn.Linear(config.decoder.hidden_size, self.vocab_size)
        # 化学分类头（预测价键/环合法性）
        self.chem_cls_head = nn.Linear(config.encoder.hidden_size + self.chem_emb_dim, 2)
        # 化学回归头（预测分子量/LogP/QED/SA
        self.chem_reg_head = nn.Linear(config.encoder.hidden_size + self.chem_emb_dim, 4)

        self.post_init() # 预训练模型参数初始化
        # 自定义参数初始化（语言建模头和化学任务头）
        init_range = getattr(config.decoder, "initializer_range", 0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=init_range)
        nn.init.normal_(self.chem_cls_head.weight, mean=0.0, std=init_range)
        nn.init.normal_(self.chem_reg_head.weight, mean=0.0, std=init_range)
        # 偏置初始化为0
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)
        nn.init.zeros_(self.chem_cls_head.bias)
        nn.init.zeros_(self.chem_reg_head.bias)

    def reparameterize(self, mu, log_var):
        """
        VAE重参数化技巧，从正态分布采样隐层向量z
        Args:
            mu: 隐层均值向量
            log_var: 隐层对数方差向量
        Returns:
            tensor: 采样得到的隐层向量z
        """
        std = torch.exp(0.5 * log_var)  # 计算标准差
        eps = torch.randn_like(std)     # 采样标准正态分布噪声
        return mu + eps * std           # 重参数化采样

    def encode(self, input_ids, attention_mask, chem_attrs):
        """
        模型编码阶段：将输入序列和化学属性编码为隐层均值和方差
        Args:
            input_ids: 输入token ID序列
            attention_mask: 注意力掩码
            chem_attrs: 归一化化学属性张量
        Returns:
            tuple: 池化特征、隐层均值mu、隐层对数方差log_var
        """
        # Encoder前向传播，获取隐藏状态
        encoder_outputs = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        encoder_hidden_states = encoder_outputs[0]

        # 化学属性嵌入并扩展到序列长度
        chem_emb = self.chem_emb_layer(chem_attrs).unsqueeze(1)
        chem_emb_repeat = chem_emb.repeat(1, encoder_hidden_states.shape[1], 1)
        # 融合Encoder隐藏状态和化学属性嵌入
        encoder_hidden_fused = torch.cat([encoder_hidden_states, chem_emb_repeat], dim=-1)

        # 自注意力融合化学属性和序列特征
        attn_output, _ = self.chem_attention(encoder_hidden_fused, encoder_hidden_fused, encoder_hidden_fused)

        # 基于注意力掩码的池化（只对有效token区域池化）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(attn_output.size()).float()
            pooled = torch.sum(attn_output * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        else:
            pooled = attn_output.mean(dim=1)  # 无掩码则全局平均池化

        # 映射到隐层均值和方差
        return pooled, self.fc_mu(pooled), self.fc_var(pooled)

    def forward(
            self, input_ids=None, attention_mask=None, labels=None,
            valence_ok=None, ring_ok=None, mw=None, logp=None, qed=None, sa=None,
            temperature=1.0, return_dict=True, **kwargs
    ):
        """
        模型前向传播，计算多任务总损失（重构+KL+化学分类+化学回归）
        Args:
            input_ids: 输入token ID序列
            attention_mask: 注意力掩码
            labels: 训练标签（与input_ids一致）
            valence_ok: 价键合法性标签
            ring_ok: 环合法性标签
            mw: 归一化分子量
            logp: 归一化LogP
            qed: QED评分
            sa: 归一化SA评分
            temperature: 采样温度（未使用）
            return_dict: 是否返回字典格式输出
            **kwargs: 其他参数
        Returns:
            Seq2SeqLMOutput/tuple: 模型输出，包含总损失和各类logits
        """
        self.train()  # 切换到训练模式
        # 拼接并归一化化学属性为4维张量
        chem_attrs = torch.stack([
            (mw - 200) / 300,
            (logp + 2) / 7,
            qed,
            (sa - 5) / 5
        ], dim=-1)

        # 编码得到隐层均值、方差，采样得到z
        pooled, mu, log_var = self.encode(input_ids, attention_mask, chem_attrs)
        z = self.reparameterize(mu, log_var)
        z_proj = self.fc_z(z).unsqueeze(1) # 映射为Decoder初始输入

        # 构建Decoder输入（标签左移一位，自回归训练）
        decoder_input_ids = labels[:, :-1] if labels is not None else input_ids
        decoder_attention_mask = attention_mask[:, :-1] if (
                    labels is not None and attention_mask is not None) else attention_mask
        # 兜底处理全0注意力掩码
        if decoder_attention_mask is not None and (decoder_attention_mask.sum() == 0):
            decoder_attention_mask = torch.ones_like(decoder_input_ids)

        # Decoder前向传播，融合隐层向量z
        decoder_outputs = self.decoder_model(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=z_proj.expand(-1, decoder_input_ids.shape[1], -1),
            encoder_attention_mask=torch.ones_like(
                decoder_attention_mask) if decoder_attention_mask is not None else None,
            return_dict=False
        )
        decoder_hidden_states = decoder_outputs[0]
        logits = self.lm_head(decoder_hidden_states)  # 计算语言建模logits

        # 1. 计算重构损失（只计算非PAD区域）
        recon_loss = torch.tensor(0.0, device=logits.device)
        if labels is not None:
            ignore_idx = 0  # PAD token ID，计算损失时忽略
            logits_reshaped = logits.reshape(-1, self.vocab_size)
            labels_reshaped = labels[:, 1:].reshape(-1)     # 标签右移一位
            non_pad_mask = labels_reshaped != ignore_idx    # 非PAD区域掩码
            if non_pad_mask.sum() > 0:
                recon_loss = F.cross_entropy(logits_reshaped[non_pad_mask], labels_reshaped[non_pad_mask])
            else:
                recon_loss = torch.tensor(1.0, device=logits.device)  # 无有效token时设置默认损失

        # 2. 计算KL散度损失（正则化隐层分布接近标准正态分布）
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / input_ids.shape[0]

        # 3. 计算化学分类损失（价键/环合法性）和回归损失（分子量/LogP/QED/SA）
        chem_cls_loss = torch.tensor(0.0, device=logits.device)
        chem_reg_loss = torch.tensor(0.0, device=logits.device)
        if valence_ok is not None and ring_ok is not None:
            cls_logits = self.chem_cls_head(pooled)
            cls_labels = torch.stack([valence_ok, ring_ok], dim=-1)
            chem_cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_labels)  # 二分类损失

            reg_logits = self.chem_reg_head(pooled)
            reg_labels = torch.stack([(mw - 200) / 300, (logp + 2) / 7, qed, (sa - 5) / 5], dim=-1)
            chem_reg_loss = F.mse_loss(reg_logits, reg_labels)  # 回归损失

        # 多任务损失加权求和（提升化学任务损失权重至0.1，增强类药性约束
        loss = recon_loss + 0.5 * kl_loss + 0.1 * chem_cls_loss + 0.1 * chem_reg_loss

        if return_dict:
            return Seq2SeqLMOutput(
                loss=loss, logits=logits, encoder_hidden_states=z, decoder_hidden_states=decoder_hidden_states
            )
        return (loss, logits) if loss is not None else logits

    def generate(self, num_samples=1000, device="cuda", temperature=0.5, max_length=60, tokenizer=None):
        """
        基于隐空间采样生成类药分子SELFIES序列，内置类药性约束（QED≥0.35、SA≤8.5）
        Args:
            num_samples: 生成样本数量
            device: 生成使用的设备（cuda/cpu）
            temperature: 采样温度（值越高多样性越强）
            max_length: 生成序列最大长度
            tokenizer: SELFIESTokenizer实例，用于解码
        Returns:
            list: 生成的有效SELFIES序列列表（已去重）
        """
        self.eval()                 # 切换到评估模式
        generated_selfies = []
        batch_size = 64             # 批量生成，避免显存溢出
        decode_fail_count = 0       # 解码失败计数
        filter_fail_count = 0       # 化学/类药性过滤失败计数

        # 定义禁止生成的SELFIES token（含剧毒/罕见原子）
        forbidden_selfies_tokens = ['[P]', '[As]', '[Se]', '[Te]', '[Branch4]']
        forbidden_token_ids = [tokenizer.vocab[tok] for tok in forbidden_selfies_tokens if tok in tokenizer.vocab]

        # 特殊token ID兜底（避免tokenizer属性缺失）
        eos_token_id = tokenizer.eos_id if hasattr(tokenizer, "eos_id") else 2
        decoder_start_token_id = tokenizer.sos_id if hasattr(tokenizer, "sos_id") else 1
        self.config.eos_token_id = eos_token_id
        self.config.decoder_start_token_id = decoder_start_token_id

        with torch.no_grad():  # 禁用梯度计算，节省显存
            # 按批次生成样本
            for i in range(0, num_samples, batch_size):
                curr_batch_size = min(batch_size, num_samples - i)
                # 从标准正态分布采样隐层向量z
                z = torch.randn(curr_batch_size, self.latent_dim, device=device)
                z_proj = self.fc_z(z).unsqueeze(1)  # 映射为Decoder初始输入
                # 初始化生成序列为SOS token
                generated_ids = torch.full((curr_batch_size, 1), decoder_start_token_id, dtype=torch.long,
                                           device=device)

                # 逐token自回归生成
                for step in range(min(max_length - 1, 40)):
                    # Decoder前向传播
                    decoder_outputs = self.decoder_model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids),
                        encoder_hidden_states=z_proj.expand(-1, generated_ids.shape[1], -1),
                        encoder_attention_mask=torch.ones_like(generated_ids),
                        return_dict=False
                    )
                    last_token_hidden = decoder_outputs[0][:, -1, :]                            # 取最后一个token的隐藏状态
                    last_token_logits = self.lm_head(last_token_hidden) / temperature           # 温度缩放logits

                    # 禁止生成非法token（降低对应logits得分）
                    if forbidden_token_ids:
                        last_token_logits[:, forbidden_token_ids] -= 3.0
                    # 多项式采样下一个token
                    next_token = torch.multinomial(F.softmax(last_token_logits, dim=-1), num_samples=1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)       # 拼接新token

                    # 每5步校验一次化学合理性，提前终止无效序列
                    if step % 5 == 0:
                        for idx in range(curr_batch_size):
                            curr_selfies = tokenizer.decode(generated_ids[idx], skip_special_tokens=True)
                            try:
                                curr_smiles = sf.decoder(curr_selfies)
                                if not filter_unreasonable_smiles(curr_smiles):
                                    generated_ids[idx, step + 2:] = eos_token_id                # 后续填充EOS，终止生成
                            except:
                                generated_ids[idx, step + 2:] = eos_token_id

                    # 若所有序列都生成了EOS，提前终止当前批次
                    if (next_token == eos_token_id).all():
                        break

                # 解码生成的ID序列为SELFIES，并过滤有效分子
                for ids in generated_ids:
                    selfies = tokenizer.decode(ids, skip_special_tokens=True)
                    if not selfies or len(selfies) < 5:                                         # 过滤过短序列
                        decode_fail_count += 1
                        continue
                    try:
                        smiles = sf.decoder(selfies)                                            # SELFIES转SMILES
                        if not smiles:
                            decode_fail_count += 1
                            continue
                        # 化学合理性过滤
                        if filter_unreasonable_smiles(smiles):
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                # 类药性初筛（放宽阈值提升生成效率
                                qed_score = QED.qed(mol)
                                sa_score = calculate_SA_score(mol)
                                if qed_score >= 0.35 and sa_score <= 8.5:
                                    generated_selfies.append(selfies)
                                else:
                                    filter_fail_count += 1
                            else:
                                filter_fail_count += 1
                        else:
                            filter_fail_count += 1
                    except Exception as e:
                        decode_fail_count += 1
                        if decode_fail_count <= 5:          # 仅输出前5个解码错误
                            print(f"解码失败：selfies={selfies[:50]}，错误={str(e)[:50]}")
                        continue

        # 输出生成调试统计
        print(f"\n生成调试统计：")
        print(f"- 解码失败数：{decode_fail_count}")
        print(f"- 化学/类药性过滤失败数：{filter_fail_count}")
        return list(set(generated_selfies))         # 去重后返回


# ====================== 七维度评估函数（放宽类药性判定阈值） ======================
#   实现分子生成质量的多维度评估和可视化，包含化学合理性、新颖性、多样性等指标
FINGERPRINT_NBITS = 256       # 分子指纹维度
FINGERPRINT_RADIUS = 1        # 分子指纹半径

def calculate_fingerprint(smiles):
    """
    计算分子的Morgan指纹（稠密格式），用于相似性和新颖性评估
    Args:
        smiles: 分子SMILES字符串
    Returns:
        np.ndarray/None: 256维稠密指纹数组，无效分子返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 计算稀疏Morgan指纹
    sparse_fp = AllChem.GetHashedMorganFingerprint(mol, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_NBITS)
    # 转换为稠密数组
    dense_fp = np.zeros(FINGERPRINT_NBITS, dtype=np.int32)
    for idx, count in sparse_fp.GetNonzeroElements().items():
        dense_fp[idx] = count
    return dense_fp

def evaluate_chemical_validity(smiles_list):
    """
    评估生成分子的化学合理性（无超价原子）
    Args:
        smiles_list: 生成的SMILES列表
    Returns:
        tuple: 化学合理性百分比、有效SMILES列表
    """
    valid_count = 0
    valid_smiles = []
    # 超价原子键数约束
    atom_constraints = {"S": 4, "P": 5, "Cl": 3, "Br": 5, "I": 6, "Se": 4, "Te": 6, "As": 5, "Si": 6, "B": 4}
    for smiles in tqdm(smiles_list, desc="1. 评估化学合理性"):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        has_supervalent = False
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            valence = atom.GetExplicitValence()
            if symbol in atom_constraints and valence > atom_constraints[symbol]:
                has_supervalent = True
                break
        if not has_supervalent:
            valid_count += 1
            valid_smiles.append(smiles)
    validity = valid_count / len(smiles_list) * 100 if len(smiles_list) > 0 else 0.0
    print(f"1. 化学合理性：{validity:.1f}%（{valid_count}/{len(smiles_list)}）")
    return validity, valid_smiles

def evaluate_novelty(generated_smiles, train_smiles, val_smiles, threshold=0.8):
    """
    评估生成分子的新颖性（与训练/验证集分子的指纹相似度低于阈值）
    Args:
        generated_smiles: 生成的SMILES列表
        train_smiles: 训练集SMILES列表
        val_smiles: 验证集SMILES列表
        threshold: 相似度阈值（低于则判定为新颖）
    Returns:
        tuple: 新颖性百分比、新颖SMILES列表
    """
    all_known_smiles = train_smiles + val_smiles
    known_fps = []
    # 计算已知分子的指纹（采样前5万条避免内存溢出）
    for smiles in tqdm(all_known_smiles[:50_000], desc="2. 计算已知指纹"):
        fp = calculate_fingerprint(smiles)
        if fp is not None:
            known_fps.append(fp)
    known_fps = np.array(known_fps)
    if len(known_fps) == 0:
        print("警告：无已知指纹，新颖性默认100%")
        return 100.0, generated_smiles
    novel_count = 0
    novel_smiles = []
    for smiles in tqdm(generated_smiles, desc="2. 评估新颖性"):
        fp = calculate_fingerprint(smiles)
        if fp is None:
            continue
        max_sim = 0.0
        # 计算与已知分子的最大Jaccard相似度
        for known_fp in known_fps:
            inter = np.sum(np.logical_and(fp, known_fp))
            union = np.sum(np.logical_or(fp, known_fp))
            sim = inter / union if union != 0 else 0.0
            if sim > max_sim:
                max_sim = sim
                if max_sim >= threshold:        # 超过阈值提前终止
                    break
        if max_sim < threshold:
            novel_count += 1
            novel_smiles.append(smiles)
    novelty = novel_count / len(generated_smiles) * 100 if len(generated_smiles) > 0 else 0.0
    print(f"2. 新颖性：{novelty:.1f}%（{novel_count}/{len(generated_smiles)}）")
    return novelty, novel_smiles

def evaluate_diversity(generated_smiles, threshold=0.7):
    """
    评估生成分子的多样性（生成分子间的平均相似度）
    Args:
        generated_smiles: 生成的SMILES列表
        threshold: 相似度阈值（无实际作用，为兼容参数）
    Returns:
        float: 多样性百分比（1-平均相似度）*100
    """
    generated_fps = []
    for smiles in tqdm(generated_smiles, desc="3. 计算生成指纹"):
        fp = calculate_fingerprint(smiles)
        if fp is not None:
            generated_fps.append(fp)
    generated_fps = np.array(generated_fps)
    if len(generated_fps) < 2:
        print("3. 多样性：0.0%（样本不足）")
        return 0.0
    similarities = []
    n = len(generated_fps)
    # 计算所有分子对的Jaccard相似度
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.sum(np.logical_and(generated_fps[i], generated_fps[j]))
            union = np.sum(np.logical_or(generated_fps[i], generated_fps[j]))
            sim = inter / union if union != 0 else 0.0
            similarities.append(sim)
    avg_sim = np.mean(similarities)
    diversity = (1 - avg_sim) * 100
    print(f"3. 多样性：{diversity:.1f}%（平均相似度{avg_sim:.3f}）")
    return diversity

def evaluate_uniqueness(raw_smiles, unique_smiles):
    """
    评估生成分子的唯一性（去重后占比）
    Args:
        raw_smiles: 去重前SMILES列表
        unique_smiles: 去重后SMILES列表
    Returns:
        float: 唯一性百分比
    """
    if len(raw_smiles) == 0:
        return 0.0
    uniqueness = len(unique_smiles) / len(raw_smiles) * 100
    print(f"4. 唯一性：{uniqueness:.1f}%（去重后{len(unique_smiles)}/去重前{len(raw_smiles)}）")
    return uniqueness

def evaluate_drug_likeness(smiles_list):
    """
    评估生成分子的类药性（Lipinski规则+QED+SA综合得分，阈值≥0.55）
    Args:
        smiles_list: 生成的SMILES列表
    Returns:
        tuple: 类药性百分比、类药SMILES列表、类药属性DataFrame
    """
    # 当前优化：放宽类药总得分阈值至0.55
    drug_like_count = 0
    drug_like_smiles = []
    lipinski_details = []
    for smiles in tqdm(smiles_list, desc="5. 计算类药性指标"):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        # 计算类药属性
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        qed_score = QED.qed(mol)
        sa_score = calculate_SA_score(mol)
        caco2_perm = 1 - (h_donors / 10 + h_acceptors / 20)  # 预测Caco2渗透性

        # 统计Lipinski规则满足数（4条）
        rules_met = 0
        rules_met += 1 if mw < 500 else 0
        rules_met += 1 if logp < 5 else 0
        rules_met += 1 if h_donors < 5 else 0
        rules_met += 1 if h_acceptors < 10 else 0

        # 计算综合类药得分（加权求和）
        lipinski_score = rules_met / 4 * 0.3            # Lipinski得分权重0.3
        qed_score_weighted = qed_score * 0.4            # QED得分权重0.4
        sa_score_weighted = (10 - sa_score) / 10 * 0.3  # SA得分权重0.3（分数越低得分越高）
        total_drug_score = lipinski_score + qed_score_weighted + sa_score_weighted

        # 记录详细属性
        lipinski_details.append({
            "SMILES": smiles,
            "分子量(MW)": round(mw, 2),
            "疏水性(LogP)": round(logp, 2),
            "氢键供体": h_donors,
            "氢键受体": h_acceptors,
            "QED评分": round(qed_score, 3),
            "合成可及性(SAS)": round(sa_score, 3),
            "Caco2渗透性(预测)": round(caco2_perm, 3),
            "Lipinski规则满足数": rules_met,
            "类药总得分": round(total_drug_score, 3)
        })

        # 类药判定（总得分≥0.55）
        if total_drug_score >= 0.55:
            drug_like_count += 1
            drug_like_smiles.append(smiles)
    drug_likeness = drug_like_count / len(smiles_list) * 100 if len(smiles_list) > 0 else 0.0
    print(f"5. 类药性（总得分≥0.55）：{drug_likeness:.1f}%（{drug_like_count}/{len(smiles_list)}）")
    lipinski_df = pd.DataFrame(lipinski_details)
    lipinski_df.to_csv("lipinski_drug_likeness_analysis_full_data.csv", index=False)
    return drug_likeness, drug_like_smiles, lipinski_df


def plot_top_molecules(smiles_list, lipinski_df, n=5, save_path="top_drug_like_molecules_full_data.png",
                       MATPLOTLIB_AVAILABLE=True):
    """
    绘制前n个类药性+新颖分子的结构图，支持Jupyter展示和本地保存
    Args:
        smiles_list: 类药SMILES列表
        lipinski_df: 类药属性DataFrame
        n: 展示的分子数量
        save_path: 图片保存路径
        MATPLOTLIB_AVAILABLE: 是否有matplotlib环境
    """
    print("\n" + "=" * 60)
    print(f"绘制前{min(n, len(smiles_list))}个类药性+新颖分子结构图...")
    print("=" * 60)
    valid_mols = []
    valid_smiles = []
    # 筛选可绘制的有效分子
    for smiles in smiles_list[:n]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_mols.append(mol)
            valid_smiles.append(smiles)
            if len(valid_mols) >= n:
                break
    if not valid_mols:
        print("⚠️  没有找到可绘制的有效分子")
        return

    try:
        legends = []
        for s in valid_smiles:
            row = lipinski_df[lipinski_df['SMILES'] == s].iloc[0]
            display_smiles = s[:40] + "..." if len(s) > 40 else s
            # 构建分子图例（包含关键类药指标）
            legends.append(
                f"SMILES: {display_smiles}\n"
                # 关键修复：满足规则数 → Lipinski规则满足数
                f"Lipinski规则: {row['Lipinski规则满足数']}/4\n"
                f"QED: {row['QED评分']:.3f}\n"
                f"SAS: {row['合成可及性(SAS)']:.3f}"
            )
        # 生成分子网格图
        img = Draw.MolsToGridImage(
            valid_mols, molsPerRow=min(len(valid_mols), 2),
            subImgSize=(400, 400), legends=legends, useSVG=False
        )
        # Jupyter展示图片
        if isinstance(img, Image.Image):
            display(img)
        else:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            pil_img = Image.open(img_byte_arr)
            display(pil_img)
        print(f"✅ Jupyter已展示 {len(valid_mols)} 个目标分子结构图")

        # 保存图片到本地
        if isinstance(img, Image.Image):
            img.save(save_path)
        else:
            pil_img.save(save_path)
        print(f"✅ 分子结构图已保存至：{save_path}")

        # matplotlib二次展示（如需）
        if MATPLOTLIB_AVAILABLE:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            plt.figure(figsize=(15, 10))
            img_plot = mpimg.imread(save_path)
            plt.imshow(img_plot)
            plt.axis("off")
            plt.title("Top Drug-like & Novel Molecules (QED+SAS Optimized)", fontsize=16, pad=20)
            plt.show()
    except Exception as e:
        print(f"❌ 绘图异常：{str(e)} | 已降级为单分子展示")
        for idx, mol in enumerate(valid_mols[:2]):
            mol_img = Draw.MolToImage(mol, size=(300, 300))
            display(Image.fromarray(mol_img))