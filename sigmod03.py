import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from Levenshtein import distance as lev_distance
import umap.umap_ as umap
import time
import itertools
import logging
from collections import Counter
from datasketch import MinHash, MinHashLSH
import concurrent.futures
from scipy.spatial import distance
from scipy.optimize import minimize
import math
import json
import statsmodels.api as sm
from collections import defaultdict
import matplotlib as mpl
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import cupy as cp
from numba import cuda, jit
import psutil
import random
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置选项（移动到所有函数之前）
SIMILARITY_METHOD = 'levenshtein'  # 'levenshtein' 或 'immunobert'
USE_GPU = True  # 是否使用GPU加速
BATCH_SIZE = 512  # 批量处理大小
SIMILARITY_THRESHOLD = 0.7  # 相似性阈值

# 1. 文件读取和预处理 - 增强错误处理和缺失值处理
def load_and_preprocess(file_path):
    """加载并预处理BCR/TCR序列文件"""
    try:
        logging.info(f"Loading file: {file_path}")
        
        # 读取数据时指定列类型
        df = pd.read_csv(file_path, sep='\t', header=0, 
                        dtype={'Protein': str, 'GeneV': str, 'GeneJ': str})
        
        # 检查必要列是否存在
        required_columns = ['Protein', 'Percent', 'Total', 'GeneV', 'GeneJ', 'annotation']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logging.warning(f"Missing columns in file: {missing_cols}. Creating with default values.")
            for col in missing_cols:
                if col == 'Protein':
                    df[col] = ''  # 关键列不能为空
                else:
                    df[col] = np.nan
        
        # 重命名列以匹配后续处理
        df = df.rename(columns={
            'Protein': 'sequence',
            'Percent': 'frequency',
            'GeneV': 'v_gene',
            'GeneJ': 'j_gene',
            'annotation': 'antigen_specificity'
        })
        
        # 处理缺失值
        # 抗原特异性处理：将NaN替换为"NAN"，保留"NO"不变
        df['antigen_specificity'] = df['antigen_specificity'].fillna('NAN')
        
        # 序列处理：空序列设为"INVALID"
        df['sequence'] = df['sequence'].fillna('')
        df.loc[df['sequence'].str.strip() == '', 'sequence'] = 'INVALID'
        
        # 频率处理：缺失频率设为中位数
        median_freq = df['frequency'].median() if not df['frequency'].empty else 0
        df['frequency'] = df['frequency'].fillna(median_freq)
        
        # 总数处理：缺失总数设为10^6
        df['Total'] = df['Total'].fillna(1000000)
        
        # 添加序列长度特征
        df['cdr3_length'] = df['sequence'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # 计算绝对频率 = 百分比 * 总序列数
        df['absolute_frequency'] = df['frequency'] * df['Total'] / 100
        
        # 过滤低频序列
        if not df.empty:
            quantile_val = df['absolute_frequency'].quantile(0.25)
            df = df[df['absolute_frequency'] > quantile_val]
        
        logging.info(f"Loaded data with {len(df)} sequences after filtering")
        return df
    
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        return pd.DataFrame()

# GPU加速的Levenshtein距离计算
@cuda.jit
def gpu_levenshtein_kernel(seqs1, seqs2, results):
    i, j = cuda.grid(2)
    if i < seqs1.shape[0] and j < seqs2.shape[0]:
        s1 = seqs1[i]
        s2 = seqs2[j]
        len1 = len(s1)
        len2 = len(s2)
        
        # 创建距离矩阵
        d = cuda.local.array((101, 101), dtype=np.int32)
        
        for x in range(len1 + 1):
            d[x, 0] = x
        for y in range(len2 + 1):
            d[0, y] = y
            
        for x in range(1, len1 + 1):
            for y in range(1, len2 + 1):
                cost = 0 if s1[x-1] == s2[y-1] else 1
                d[x, y] = min(
                    d[x-1, y] + 1,
                    d[x, y-1] + 1,
                    d[x-1, y-1] + cost
                )
        
        max_len = max(len1, len2)
        if max_len == 0:
            results[i, j] = 0.0
        else:
            results[i, j] = 1 - d[len1, len2] / max_len

def gpu_levenshtein_batch(seqs1, seqs2):
    """使用GPU批量计算Levenshtein相似度"""
    # 转换为固定长度的字符数组
    max_len = max(max(len(s) for s in seqs1), max(len(s) for s in seqs2))
    seqs1_arr = np.array([s.ljust(max_len) for s in seqs1], dtype='S')
    seqs2_arr = np.array([s.ljust(max_len) for s in seqs2], dtype='S')
    
    # 分配GPU内存
    d_seq1 = cuda.to_device(seqs1_arr)
    d_seq2 = cuda.to_device(seqs2_arr)
    d_results = cuda.device_array((len(seqs1), len(seqs2)), dtype=np.float32)
    
    # 配置内核
    threads_per_block = (16, 16)
    blocks_per_grid_x = (len(seqs1) + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (len(seqs2) + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # 启动内核
    gpu_levenshtein_kernel[blocks_per_grid, threads_per_block](d_seq1, d_seq2, d_results)
    
    # 将结果复制回主机
    return d_results.copy_to_host()

# ImmunoBERT嵌入计算 - 使用替代模型
class ImmunoBERTEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # 尝试加载不同的模型
        model_options = [
            "wukevin/tcr-bert",  # TCR-BERT模型
            "wukevin/tcr-bert-mlm-only",  # TCR-BERT MLM模型
            "wukevin/foldingdiff_cath",  # 蛋白质折叠模型
            "facebook/esm2_t6_8M_UR50D"  # ESM模型作为回退
        ]
        
        for model_name in model_options:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(device)
                self.model.eval()
                logging.info(f"Successfully loaded model: {model_name}")
                break  # 成功加载模型后跳出循环
            except Exception as e:
                logging.warning(f"Could not load model {model_name}: {str(e)}")
                continue
        
        # 如果所有模型都加载失败，使用随机初始化的模型作为最后手段
        if self.model is None:
            logging.error("All model loading attempts failed. Using random embeddings.")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            self.model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
            self.model.eval()
            logging.warning("Using ESM model with random weights as fallback")
    
    def embed_sequences(self, sequences, batch_size=32):
        """计算序列的嵌入"""
        embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch_seqs, 
                    padding=True, 
                    truncation=True, 
                    max_length=64, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS]标记作为整个序列的表示
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
            except Exception as e:
                logging.error(f"Error embedding sequences: {str(e)}")
                # 如果嵌入失败，使用随机向量作为占位符
                placeholder = np.random.randn(len(batch_seqs), 768)  # 假设维度为768
                embeddings.append(placeholder)
        
        return np.vstack(embeddings)

# 序列相似性计算 - 支持多种方法
def calculate_similarity_matrix(seqs1, seqs2, method='levenshtein', embedder=None):
    if method == 'levenshtein':
        if USE_GPU and cuda.is_available():
            return gpu_levenshtein_batch(seqs1, seqs2)
        else:
            # 创建维度变量提高可读性
            dim1 = len(seqs1)
            dim2 = len(seqs2)
            sim_matrix = np.zeros((dim1, dim2))  # 正确的括号嵌套
            for i, s1 in enumerate(seqs1):
                for j, s2 in enumerate(seqs2):
                    sim_matrix[i, j] = calculate_similarity(s1, s2)
            return sim_matrix
    elif method == 'immunobert' and embedder is not None:
        # 计算ImmunoBERT嵌入
        embeddings1 = embedder.embed_sequences(seqs1)
        embeddings2 = embedder.embed_sequences(seqs2)
        
        # 计算余弦相似度
        return cosine_similarity(embeddings1, embeddings2)
    else:
        logging.warning("Invalid similarity method or missing embedder. Using Levenshtein.")
        # 创建维度变量提高可读性
        dim1 = len(seqs1)
        dim2 = len(seqs2)
        sim_matrix = np.zeros((dim1, dim2))  # 正确的括号嵌套
        for i, s1 in enumerate(seqs1):
            for j, s2 in enumerate(seqs2):
                sim_matrix[i, j] = calculate_similarity(s1, s2)
        return sim_matrix

def calculate_similarity(seq1, seq2):
    """使用Levenshtein距离计算序列相似性 - 更高效"""
    if not seq1 or not seq2 or seq1 == 'INVALID' or seq2 == 'INVALID':
        return 0.0
    
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    
    # 计算标准化Levenshtein距离 (1 - 编辑距离/最大长度)
    return 1 - lev_distance(seq1, seq2) / max_len

# 动态批处理机制 - 增强内存管理
def adaptive_batching(group_seqs, max_mem_usage=0.8):
    """根据可用内存动态调整批处理大小 - 针对10k+序列优化"""
    if not group_seqs:
        return 64  # 默认值
    
    # 估计单个序列的内存使用量（更精确的估算）
    seq_len = max(len(s) for s in group_seqs)
    seq_mem = seq_len * 8  # 使用8字节浮点数
    
    # 获取可用内存
    mem_available = psutil.virtual_memory().available * max_mem_usage
    
    # 考虑GPU内存或CPU内存
    if USE_GPU and cuda.is_available():
        try:
            gpu_mem = cp.cuda.Device().mem_info[0]  # 总GPU内存
            mem_available = min(mem_available, gpu_mem * 0.7)
            logging.info(f"Using GPU memory limit: {mem_available/1e6:.2f}MB")
        except:
            pass
    
    # 计算安全批处理大小 - 使用更智能的公式
    batch_size = min(len(group_seqs), 
                   int(np.sqrt(mem_available / (seq_mem * 2))))  # 平方根关系优化大序列
    
    # 设置最小值和最大值
    batch_size = max(32, min(batch_size, 2048))
    
    logging.info(f"Adaptive batching: Available mem={mem_available/1e6:.2f}MB, " 
                f"Estimated seq mem={seq_mem} bytes, Selected batch_size={batch_size}")
    
    return batch_size

# 新增：大规模计算优化 - MinHash LSH预过滤
def minhash_prefilter(group_seqs, threshold=0.7):
    """使用MinHash进行相似序列预过滤 - 针对10k+序列优化"""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)  # 增加排列数量提高精度
    minhashes = {}
    
    for idx, seq in enumerate(group_seqs):
        m = MinHash(num_perm=128)
        for j in range(0, len(seq), 3):  # 步长为3的分词
            token = seq[j:j+3]
            m.update(token.encode('utf8'))
        minhashes[idx] = m
        lsh.insert(idx, m)
    
    candidate_pairs = set()
    for idx in minhashes:
        result = lsh.query(minhashes[idx])
        for candidate in result:
            if candidate > idx:  # 避免重复
                candidate_pairs.add((idx, candidate))
    
    return candidate_pairs

# 图不变计算 - 增强免疫学解释
def calculate_graph_invariants(G):
    """计算图的不变量并添加免疫学解释"""
    invariants = {
        'spectral_gap': np.nan,
        'girth': np.nan,
        'chromatic_number': np.nan,
        'diameter': np.nan,
        'average_path_length': np.nan
    }
    
    interpretation = {
        'spectral_gap': "Larger spectral gaps indicate tighter immune network clustering",
        'girth': "Smaller girth suggests potential cross-reactivity in immune responses",
        'chromatic_number': "Relates to the minimal number of antigen groups needed to color clones without conflict",
        'diameter': "Network diameter represents the maximal path between any two clonotypes",
        'average_path_length': "Shorter paths indicate more efficient information dissemination in the immune system"
    }
    
    if G.number_of_nodes() == 0:
        return invariants, interpretation
    
    # 谱间隙（使用Fiedler值） - 使用稀疏矩阵提高效率
    try:
        if G.number_of_nodes() > 1000:
            # 使用稀疏矩阵提高效率
            laplacian = nx.laplacian_matrix(G)
            eigenvalues, _ = eigs(laplacian, k=min(5, len(G.nodes)-1), which='SM')
            eigenvalues = np.real(eigenvalues)
            eigenvalues.sort()
            invariants['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        else:
            laplacian = nx.laplacian_matrix(G).todense()
            eigenvalues = np.linalg.eigvalsh(laplacian)
            invariants['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    except Exception as e:
        logging.error(f"Spectral gap calculation error: {str(e)}")
    
    # 周长（使用近似方法） - 更高效的近似算法
    try:
        invariants['girth'] = nx.approximation.girth(G, k=5)[0]  # 使用k=5提高大图效率
    except:
        try:
            invariants['girth'] = nx.girth(G)
        except Exception as e:
            logging.error(f"Girth calculation error: {str(e)}")
    
    # 染色数 - 使用贪心算法并添加免疫学解释
    try:
        # 使用贪心着色算法并计算使用的颜色数量
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        chromatic_num = max(coloring.values()) + 1  # 颜色索引从0开始，所以+1
        invariants['chromatic_number'] = chromatic_num
        interpretation['chromatic_number'] = f"Chromatic number {chromatic_num} indicates distinct functional groups in the immune repertoire"
    except Exception as e:
        logging.error(f"Chromatic number calculation error: {str(e)}")
    
    # 直径和平均路径长度（仅连通图） - 使用更高效的算法
    if nx.is_connected(G):
        try:
            # 使用近似方法处理大图
            if G.number_of_nodes() > 1000:
                invariants['diameter'] = nx.approximation.diameter(G)
                invariants['average_path_length'] = nx.approximation.average_shortest_path_length(G)
            else:
                invariants['diameter'] = nx.diameter(G)
                invariants['average_path_length'] = nx.average_shortest_path_length(G)
        except Exception as e:
            logging.error(f"Diameter/APL calculation error: {str(e)}")
    
    return invariants, interpretation

# 新增：免疫学特异性指标计算
def calculate_immunological_metrics(G):
    """计算免疫学相关指标 - 将图论结果转化为免疫学洞见"""
    metrics = {
        'shannon_diversity': 0,
        'v_gene_usage': {},
        'j_gene_usage': {},
        'cdr3_length_distribution': {},
        'antigen_specificity_distribution': {},
        'cross_reactivity_index': 0,
        'interpretation': {}
    }
    
    if G.number_of_nodes() == 0:
        logging.warning("Graph has no nodes - returning default metrics")
        return metrics
    
    # 1. Shannon多样性指数 - 抗原特异性
    antigen_counts = Counter(nx.get_node_attributes(G, 'antigen_specificity').values())
    total_nodes = sum(antigen_counts.values())
    
    # 过滤掉无效抗原
    valid_antigens = {ag: count for ag, count in antigen_counts.items() if ag not in ['NAN', 'NO']}
    valid_count = sum(valid_antigens.values())
    
    if valid_count > 0:
        shannon = 0
        for count in valid_antigens.values():
            p = count / valid_count
            shannon -= p * math.log(p) if p > 0 else 0
        
        metrics['shannon_diversity'] = shannon
        metrics['interpretation']['shannon_diversity'] = f"Diversity index {shannon:.3f} indicates antigen-specific repertoire heterogeneity"
    else:
        metrics['interpretation']['shannon_diversity'] = "No valid antigen specificity data for diversity calculation"
    
    # 2. V/J基因使用频率
    v_genes = Counter(nx.get_node_attributes(G, 'v_gene').values())
    j_genes = Counter(nx.get_node_attributes(G, 'j_gene').values())
    
    metrics['v_gene_usage'] = {gene: count/total_nodes for gene, count in v_genes.items()}
    metrics['j_gene_usage'] = {gene: count/total_nodes for gene, count in j_genes.items()}
    
    # 3. CDR3长度分布
    cdr3_lengths = [data['cdr3_length'] for node, data in G.nodes(data=True)]
    if cdr3_lengths:
        metrics['cdr3_length_distribution'] = {
            'mean': np.mean(cdr3_lengths),
            'std': np.std(cdr3_lengths),
            'min': min(cdr3_lengths),
            'max': max(cdr3_lengths)
        }
        metrics['interpretation']['cdr3_length'] = (
            f"CDR3 lengths average {metrics['cdr3_length_distribution']['mean']:.1f} AA, "
            f"range {min(cdr3_lengths)}-{max(cdr3_lengths)}"
        )
    else:
        metrics['interpretation']['cdr3_length'] = "No CDR3 length data available"
    
    # 4. 抗原特异性分布
    metrics['antigen_specificity_distribution'] = antigen_counts
    if antigen_counts:
        dominant_antigen = max(antigen_counts, key=antigen_counts.get)
        metrics['interpretation']['antigen'] = (
            f"Dominant antigen specificity: {dominant_antigen} "
            f"({antigen_counts[dominant_antigen]/total_nodes:.1%})"
        )
    else:
        metrics['interpretation']['antigen'] = "No antigen specificity data available"
    
    # 5. 交叉反应指数 - 不同抗原组间的连接（增强版本）
    cross_edges = 0
    total_valid_edges = 0  # 只考虑有效抗原对
    
    if G.number_of_edges() > 0:
        for u, v, data in G.edges(data=True):
            u_antigen = G.nodes[u].get('antigen_specificity', 'NAN')
            v_antigen = G.nodes[v].get('antigen_specificity', 'NAN')
            
            # 跳过无效抗原
            if u_antigen in ['NAN', 'NO'] or v_antigen in ['NAN', 'NO']:
                continue
                
            total_valid_edges += 1
            
            # 统计不同抗原间的连接
            if u_antigen != v_antigen:
                cross_edges += 1
        
        # 计算比例 - 确保分母不为零
        if total_valid_edges > 0:
            metrics['cross_reactivity_index'] = cross_edges / total_valid_edges
            metrics['interpretation']['cross_reactivity'] = (
                f"Cross-reactivity index {metrics['cross_reactivity_index']:.3f} "
                "indicates potential for immune cross-reactivity"
            )
        else:
            metrics['interpretation']['cross_reactivity'] = (
                "No valid antigen pairs found - cannot calculate cross-reactivity"
            )
            
        # 添加调试日志
        logging.info(
            f"Cross-reactivity calculation: {cross_edges} cross-edges / "
            f"{total_valid_edges} valid edges = {metrics['cross_reactivity_index']:.3f}"
        )
    else:
        metrics['interpretation']['cross_reactivity'] = "No edges in graph - cross-reactivity not applicable"
    
    return metrics

# 算法公平性分析 - 增强免疫学相关性
def analyze_algorithmic_fairness(G):
    """
    检验不同抗原组在网络中的表示公平性
    添加免疫学相关解释
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    
    antigen_groups = defaultdict(list)
    for node in G.nodes:
        antigen = G.nodes[node]['antigen_specificity']
        antigen_groups[antigen].append(node)
    
    centrality_scores = []
    for antigen, nodes in antigen_groups.items():
        if antigen in ['NAN', 'NO']:
            continue
            
        # 计算组内的平均中心性
        group_centrality = np.mean([G.nodes[n]['centrality'] for n in nodes])
        centrality_scores.append({
            'antigen': antigen,
            'group_size': len(nodes),
            'mean_centrality': group_centrality,
            'centrality_std': np.std([G.nodes[n]['centrality'] for n in nodes]),
            'fraction_central': np.mean([G.nodes[n]['centrality'] > 0.5 for n in nodes]),
            'immunological_interpretation': f"Higher centrality suggests antigen {antigen} has prominent immune recognition"
        })
    
    return pd.DataFrame(centrality_scores)

# 信息传播模拟 - 添加免疫学解释
def simulate_information_diffusion(G, seeds, steps=10):
    """
    模拟抗原特异性信号在网络中的传播
    添加免疫学相关输出
    """
    if G.number_of_nodes() == 0 or not seeds:
        return set(), ""
    
    infected = set(seeds)
    antigen_spread = Counter()
    for seed in seeds:
        antigen_spread[G.nodes[seed]['antigen_specificity']] += 1
    
    interpretation = ""
    if antigen_spread:
        dominant_antigen = max(antigen_spread, key=antigen_spread.get)
        interpretation = f"Initial signal dominated by {dominant_antigen} antigen specificity"
    
    for _ in range(steps):
        new_infected = set()
        for node in infected:
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in infected:
                    # 传播概率基于边的权重（相似性）
                    weight = G[node][neighbor].get('weight', 0)
                    if random.random() < weight: 
                        new_infected.add(neighbor)
                        antigen_spread[G.nodes[neighbor]['antigen_specificity']] += 1
        infected |= new_infected
    
    # 添加免疫学解释
    if len(infected) > 0:
        final_antigens = Counter(G.nodes[n]['antigen_specificity'] for n in infected)
        dominant_final = max(final_antigens, key=final_antigens.get)
        interpretation += f"\nFinal network coverage: {len(infected)/len(G.nodes):.1%}, " \
                        f"dominated by {dominant_final} specificity"
    
    return infected, interpretation

# 3. 网络构建 (MetaNet方法) - 针对10k+序列优化
def build_metanet(df, similarity_threshold=SIMILARITY_THRESHOLD):
    """构建BCR/TCR相似性网络 - 针对大规模数据优化"""
    global SIMILARITY_METHOD
    
    logging.info(f"Building similarity network using {SIMILARITY_METHOD} method...")
    
    if df.empty:
        logging.warning("Empty dataframe, returning empty graph")
        return nx.Graph()
    
    G = nx.Graph()
    embedder = None
    if SIMILARITY_METHOD == 'immunobert':
        device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        try:
            embedder = ImmunoBERTEmbedder(device=device)
        except Exception as e:
            logging.error(f"Error initializing ImmunoBERTEmbedder: {str(e)}. Falling back to Levenshtein method.")
            SIMILARITY_METHOD = 'levenshtein'
    
    # 添加节点 - 确保所有行使用相同的缩进（4个空格）
    for idx, row in df.iterrows():
        sequence = row['sequence']
        frequency = float(row['absolute_frequency'])
        v_gene = str(row['v_gene'])
        j_gene = str(row['j_gene'])  # 确保这行使用与其他行相同的缩进
        cdr3_length = int(row['cdr3_length'])
        antigen = str(row['antigen_specificity'])
        
        G.add_node(
            sequence,
            frequency=frequency,
            v_gene=v_gene,
            j_gene=j_gene,
            cdr3_length=cdr3_length,
            antigen_specificity=antigen
        )
    
    # 添加边（基于序列相似性） - 使用V/J基因分组优化
    vj_groups = df[['v_gene', 'j_gene']].drop_duplicates()
    
    # 进度计数器
    total_comparisons = 0
    for _, group in vj_groups.iterrows():
        v_gene = str(group['v_gene'])
        j_gene = str(group['j_gene'])
        group_seqs = df[
            (df['v_gene'].astype(str) == v_gene) & 
            (df['j_gene'].astype(str) == j_gene)
        ]['sequence'].tolist()
        group_size = len(group_seqs)
        total_comparisons += group_size * (group_size - 1) // 2
    
    completed = 0
    last_percent = -1
    
    # 批量处理组 - 添加MinHash预过滤
    for _, group in vj_groups.iterrows():
        v_gene = str(group['v_gene'])
        j_gene = str(group['j_gene'])
        group_seqs = df[
            (df['v_gene'].astype(str) == v_gene) & 
            (df['j_gene'].astype(str) == j_gene)
        ]['sequence'].tolist()
        group_size = len(group_seqs)
        
        if group_size < 2:
            continue
        
        # 动态调整批处理大小 - 针对大规模优化
        batch_size = adaptive_batching(group_seqs)
        
        # 对于大型组使用MinHash预过滤
        if group_size > 1000:
            logging.info(f"Using MinHash LSH prefiltering for large group: {v_gene}-{j_gene} ({group_size} seqs)")
            candidate_pairs = minhash_prefilter(group_seqs, similarity_threshold * 0.9)
            total_candidates = len(candidate_pairs)
            logging.info(f"MinHash reduced comparisons from {group_size*(group_size-1)//2} to {total_candidates}")
            
            # 分批处理候选对
            for i in range(0, total_candidates, batch_size):
                batch_pairs = list(candidate_pairs)[i:i+batch_size]
                batch1 = [group_seqs[idx1] for (idx1, idx2) in batch_pairs]
                batch2 = [group_seqs[idx2] for (idx1, idx2) in batch_pairs]
                
                # 计算批处理相似度
                if SIMILARITY_METHOD == 'immunobert' and embedder is not None:
                    # 对于ImmunoBERT，批量计算嵌入
                    all_seqs = list(set(batch1 + batch2))
                    embeddings = embedder.embed_sequences(all_seqs)
                    emb_dict = {seq: emb for seq, emb in zip(all_seqs, embeddings)}
                    
                    for (s1, s2) in zip(batch1, batch2):
                        emb1 = emb_dict[s1]
                        emb2 = emb_dict[s2]
                        sim = cosine_similarity([emb1], [emb2])[0][0]
                        if sim > similarity_threshold:
                            G.add_edge(s1, s2, weight=sim, edge_type='similarity')
                else:
                    # 对于Levenshtein
                    for s1, s2 in zip(batch1, batch2):
                        sim = calculate_similarity(s1, s2)
                        if sim > similarity_threshold:
                            G.add_edge(s1, s2, weight=sim, edge_type='similarity')
                
                completed += len(batch_pairs)
        else:
            # 对于小规模组使用原有的批量处理
            # 分批计算相似度矩阵
            for i in range(0, group_size, batch_size):
                batch1 = group_seqs[i:i+batch_size]
                for j in range(i, group_size, batch_size):
                    batch2 = group_seqs[j:j+batch_size] if j != i else group_seqs[i+1:j+batch_size]
                    
                    # 计算批处理相似度矩阵
                    sim_matrix = calculate_similarity_matrix(batch1, batch2, method=SIMILARITY_METHOD, embedder=embedder)
                    
                    # 添加边
                    for k, s1 in enumerate(batch1):
                        for l, s2 in enumerate(batch2):
                            if j == i and l <= k:  # 避免重复计算对角线
                                continue
                            if sim_matrix[k, l] > similarity_threshold:
                                G.add_edge(s1, s2, weight=sim_matrix[k, l], edge_type='similarity')
                
                completed += len(batch1) * len(group_seqs) / 2
        
        # 进度更新
        if total_comparisons > 0:
            percent_done = int((completed / total_comparisons) * 100)
            if percent_done > last_percent and percent_done % 5 == 0:
                logging.info(f"Network building progress: {percent_done}%")
                last_percent = percent_done
    
    logging.info(f"Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# 4. 网络分析和特征提取 - 增强免疫学分析
def analyze_network(G):
    """分析网络并提取特征 - 增加免疫学指标"""
    logging.info("Analyzing network features with immunological insights...")
    
    if len(G.nodes) == 0:
        logging.warning("Empty graph, returning empty features")
        return np.array([]), G, {}, {}
    
    # 计算节点度中心性
    degree_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, degree_centrality, 'centrality')
    
    # 计算特征向量中心性（如果节点数量较少）
    eigen_flag = False
    if len(G.nodes) < 500:  # 提高节点数量阈值
        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G, max_iter=1000)
            nx.set_node_attributes(G, eigenvector_centrality, 'eigen_centrality')
            eigen_flag = True
        except Exception as e:
            logging.warning(f"Eigenvector centrality calculation failed: {str(e)}")
    else:
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
            nx.set_node_attributes(G, pagerank, 'pagerank_centrality')
        except Exception as e:
            logging.warning(f"PageRank centrality calculation failed: {str(e)}")
    
    # 检测社区结构 - 增强免疫学解释
    try:
        communities = nx.community.louvain_communities(G, resolution=1.0, seed=42)
        community_antigens = []
        
        for i, comm in enumerate(communities):
            # 分析社区中的主要抗原特异性
            antigen_counts = Counter(G.nodes[n]['antigen_specificity'] for n in comm)
            dominant_antigen = antigen_counts.most_common(1)[0][0] if antigen_counts else 'UNKNOWN'
            
            # 添加社区属性
            for node in comm:
                G.nodes[node]['community'] = i
                G.nodes[node]['dominant_antigen'] = dominant_antigen
            
            community_antigens.append({
                'community_id': i,
                'size': len(comm),
                'dominant_antigen': dominant_antigen,
                'antigen_purity': antigen_counts[dominant_antigen] / len(comm) if comm else 0
            })
            
        logging.info(f"Detected {len(communities)} communities with antigen specificity")
    except Exception as e:
        logging.error(f"Community detection failed: {str(e)}")
        for node in G.nodes:
            G.nodes[node]['community'] = -1
    
    # 计算图不变量（含免疫学解释）
    invariants, invariants_interpretation = calculate_graph_invariants(G)
    
    # 计算免疫学特异性指标
    immunological_metrics = calculate_immunological_metrics(G)
    
    # 提取特征矩阵
    features = []
    for node, data in G.nodes(data=True):
        feature_row = [
            data['frequency'],
            data['cdr3_length'],
            data['centrality'],
            data.get('community', -1)
        ]
        
        if 'eigen_centrality' in data:
            feature_row.append(data['eigen_centrality'])
        elif 'pagerank_centrality' in data:
            feature_row.append(data['pagerank_centrality'])
        
        features.append(feature_row)
    
    return np.array(features), G, invariants, immunological_metrics, invariants_interpretation

# 5. 可视化功能 - 增强免疫学解释
def visualize_results(G, features, invariants, immunological_metrics, invariants_interpretation, output_dir, method_name):
    """生成可视化图表并保存，增加免疫学解释图表（交叉反应指数单独生成）"""
    logging.info("Generating visualizations with immunological insights...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查可用的样式并选择合适的
    available_styles = plt.style.available
    preferred_styles = ['seaborn-poster', 'seaborn-whitegrid', 'ggplot', 'seaborn']
    selected_style = None
    
    for style in preferred_styles:
        if style in available_styles:
            selected_style = style
            break
    
    if selected_style:
        plt.style.use(selected_style)
        logging.info(f"Using plot style: {selected_style}")
    else:
        logging.warning("No preferred plot styles available. Using default.")
        # 手动设置一些样式参数
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 8),
            'figure.dpi': 300
        })
    
    # 创建过滤后的节点列表（排除NAN）
    filtered_nodes = [n for n in G.nodes if G.nodes[n]['antigen_specificity'] != 'NAN']
    filtered_G = G.subgraph(filtered_nodes).copy()
    filtered_features = [features[i] for i, n in enumerate(G.nodes) if n in filtered_nodes]
    
    logging.info(f"Visualizing {len(filtered_nodes)} nodes (excluding NAN antigen specificity)")
    
    # 创建颜色映射
    def create_antigen_colormap(antigens):
        """创建抗原特异性颜色映射"""
        antigen_counts = Counter(antigens)
        unique_antigens = sorted(antigen_counts.keys(), key=lambda x: antigen_counts[x], reverse=True)
        
        # 为10种最常见的抗原使用清晰的配色，其余使用"其他"类别
        if len(unique_antigens) > 15:
            main_antigens = unique_antigens[:10]
            other_antigens = set(unique_antigens[10:])
            
            for node in filtered_G.nodes:
                if filtered_G.nodes[node]['antigen_specificity'] in other_antigens:
                    filtered_G.nodes[node]['antigen_specificity'] = 'OTHER'
            
            antigens = [ant if ant in main_antigens else 'OTHER' for ant in antigens]
            unique_antigens = main_antigens + ['OTHER']
        
        # 生成颜色
        antigen_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_antigens)))
        return {antigen: antigen_colors[i] for i, antigen in enumerate(unique_antigens)}, antigens
    
    # 1. 网络可视化 - 按抗原特异性着色（过滤NAN）
    plt.figure(figsize=(15, 12))
    
    if len(filtered_G.nodes) > 0:
        antigen_labels = [filtered_G.nodes[n]['antigen_specificity'] for n in filtered_G.nodes]
        color_map, updated_antigens = create_antigen_colormap(antigen_labels)
        
        # 获取位置
        if len(filtered_G.nodes) > 100:
            pos = nx.spring_layout(filtered_G, k=0.2, seed=42)
        else:
            pos = nx.kamada_kawai_layout(filtered_G)
        
        # 按抗原特异性着色
        node_colors = [color_map.get(ant, 'gray') for ant in updated_antigens]
        
        nx.draw_networkx_nodes(
            filtered_G, pos, node_size=80,
            node_color=node_colors,
            alpha=0.8
        )
        
        # 仅当边数少于500时才绘制边
        if len(filtered_G.edges) < 500:
            nx.draw_networkx_edges(
                filtered_G, pos, alpha=0.1,
                edge_color='lightgray'
            )
        
        # 添加图例
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color_map[ant], markersize=8,
                                    label=ant) 
                         for ant in color_map]
        
        plt.legend(handles=legend_patches, title='Antigen Specificity', 
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.title(f'BCR/TCR Similarity Network ({method_name})')
    
    plt.tight_layout()
    network_path = os.path.join(output_dir, f'antigen_network_{method_name}.png')
    plt.savefig(network_path, dpi=300)
    plt.close()
    logging.info(f"Created network visualization: {network_path}")
    
    # ========= 单独生成交叉反应指数图表 =========
    cross_reactivity = immunological_metrics.get('cross_reactivity_index', 0)
    
    # 创建独立图表
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    # 确保值为数值类型
    if not isinstance(cross_reactivity, (int, float)):
        logging.warning(f"Invalid cross-reactivity value: {cross_reactivity} - coercing to float")
        try:
            cross_reactivity = float(cross_reactivity)
        except (ValueError, TypeError):
            cross_reactivity = 0.0
            logging.error("Could not convert cross-reactivity to float - using 0")
    
    # 创建可见的条形图
    visible_height = max(cross_reactivity, 0.01)  # 确保最小高度可见
    bar = plt.bar(['Cross-Reactivity'], [visible_height], color='salmon')
    
    # 添加值标签
    label_y = cross_reactivity + 0.01 if cross_reactivity > 0 else visible_height + 0.01
    plt.text(0, label_y, f'{cross_reactivity:.3f}', ha='center', va='bottom', fontsize=12)
    
    # 添加无交叉反应的解释文本（如果需要）
    if cross_reactivity <= 0:
        plt.text(0.5, 0.4, 
                 "No cross-reactive edges detected\n"
                 "(All edges connect same-antigen sequences)", 
                 ha='center', 
                 va='center',
                 transform=ax.transAxes,
                 fontsize=10,
                 color='red',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title('Cross-Reactivity Index', fontsize=14)
    plt.ylabel('Fraction of edges', fontsize=12)
    
    # 设置合理的Y轴范围
    y_max = max(0.5, cross_reactivity * 1.5) if cross_reactivity > 0 else 0.5
    plt.ylim(0, y_max)
    
    plt.figtext(0.5, 0.05, "Higher values indicate greater cross-reactive potential", 
                ha='center', fontsize=10, color='dimgray')
    
    # 添加免疫学解释
    cross_interpretation = immunological_metrics.get('interpretation', {}).get('cross_reactivity', '')
    if cross_interpretation:
        plt.figtext(0.5, 0.01, cross_interpretation, 
                   ha='center', fontsize=9, style='italic')
    
    cross_path = os.path.join(output_dir, f'cross_reactivity_{method_name}.png')
    plt.tight_layout()
    plt.savefig(cross_path, dpi=300)
    plt.close()
    logging.info(f"Created standalone cross-reactivity visualization: {cross_path}")
    
    # 2. 免疫学见解可视化（修改后不再包含交叉反应指数）
    # 2. 免疫学见解可视化（修改后）
    plt.figure(figsize=(12, 10))  # 增加高度为标签提供更多空间
    
    # 1. V基因使用分布 - 优化标签布局
    v_gene_usage = immunological_metrics.get('v_gene_usage', {})
    plt.subplot(2, 2, 1)
    if v_gene_usage:
        v_genes, v_counts = zip(*sorted(v_gene_usage.items(), key=lambda x: x[1], reverse=True)[:10])
        plt.bar(v_genes, v_counts, color='lightgreen')
        plt.title('Top 10 V Gene Usage', pad=15)  # 增加标题与图的间距
        plt.xlabel('V Gene')
        plt.ylabel('Frequency')
        
        # 优化标签布局
        plt.xticks(
            rotation=45, 
            ha='right', 
            fontsize=8,  # 减小字体大小
            rotation_mode='anchor'  # 更好的旋转锚点
        )
        
        # 增加底部空间
        plt.subplots_adjust(bottom=0.15)
        
        # 移动说明文本位置
        plt.figtext(0.3, 0.09, "V gene usage reflects repertoire diversity", 
                   ha='center', fontsize=9)
    
    # 2. CDR3长度分布
    cdr3_dist = immunological_metrics.get('cdr3_length_distribution', {})
    plt.subplot(2, 2, 2)
    if cdr3_dist:
        measures = ['Mean', 'Min', 'Max']
        values = [cdr3_dist.get('mean', 0), cdr3_dist.get('min', 0), cdr3_dist.get('max', 0)]
        plt.bar(measures, values, color=['blue', 'green', 'red'])
        plt.title('CDR3 Length Distribution', pad=15)
        plt.ylabel('Length (AA)')
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, f'{v:.1f}', ha='center')
        
        # 移动说明文本位置
        plt.figtext(0.75, 0.09, "CDR3 length affects antigen recognition", 
                   ha='center', fontsize=9)
    
    # 3. Shannon多样性指数 - 优化文本位置
    diversity = immunological_metrics.get('shannon_diversity', 0)
    plt.subplot(2, 2, 3)
    plt.bar(['Diversity'], [diversity], color='skyblue')
    plt.text(0, diversity, f'{diversity:.3f}', ha='center', va='bottom')
    plt.title('Antigen Diversity Index', pad=15)
    plt.ylabel('Shannon Diversity')
    
    # 将解释文本放在图表下方
    interpretation_text = immunological_metrics['interpretation'].get('shannon_diversity', '')
    plt.figtext(0.5, 0.25, interpretation_text, 
               ha='center', fontsize=9, wrap=True)  # 使用wrap自动换行
    
    # 4. 抗原特异性分布 - 优化标签位置
    plt.subplot(2, 2, 4)
    antigen_dist = immunological_metrics.get('antigen_specificity_distribution', {})
    if antigen_dist:
        # 过滤无效抗原
        valid_antigens = {k: v for k, v in antigen_dist.items() if k not in ['NAN', 'NO']}
        if valid_antigens:
            antigens = list(valid_antigens.keys())
            counts = list(valid_antigens.values())
            
            # 对主要抗原进行分组处理
            if len(antigens) > 5:
                main_counts = sorted(valid_antigens.items(), key=lambda x: x[1], reverse=True)[:5]
                other_count = sum(v for k, v in valid_antigens.items() if k not in dict(main_counts).keys())
                antigens, counts = zip(*main_counts)
                antigens = list(antigens) + ['Other']
                counts = list(counts) + [other_count]
            
            # 添加标签百分比
            total = sum(counts)
            percentages = [f'{c/total:.1%}' if total > 0 else '0%' for c in counts]
            
            # 创建自定义标签 - 带百分比
            labels = [f"{ant}\n({pct})" for ant, pct in zip(antigens, percentages)]
            
            # 增大饼图间距，远离中心
            explode = [0.05] * len(antigens)  # 所有扇区稍微分离
            
            # 绘制饼图，调整标签位置
            wedges, texts, autotexts = plt.pie(
                counts,
                labels=labels,
                explode=explode,
                startangle=90,
                pctdistance=0.85,
                textprops={'fontsize': 8},  # 减小标签字体
                autopct='%1.1f%%',
                wedgeprops={'edgecolor': 'w', 'linewidth': 0.5}
            )
            
            # 优化标签位置以避免重叠
            for text in texts + autotexts:
                text.set_fontsize(8)  # 确保所有文本都小
            
            plt.title('Antigen Specificity Distribution', pad=15)
            plt.axis('equal')
        else:
            plt.text(0.5, 0.5, "No valid antigen data", ha='center', va='center')
    else:
        plt.text(0.5, 0.5, "No antigen distribution data", ha='center', va='center')
    
    # 增加整个图形的底部空间
    plt.subplots_adjust(
        bottom=0.15, 
        top=0.95,  # 减小顶部空间
        hspace=0.4,  # 增加行间距
        wspace=0.3   # 增加列间距
    )
    
    immuno_path = os.path.join(output_dir, f'immunological_insights_{method_name}.png')
    plt.savefig(immuno_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Created immunological insights visualization (without cross-reactivity): {immuno_path}")
    
    # 3. 图不变量可视化（含免疫学解释）
    plt.figure(figsize=(12, 6))
    
    # 创建解释文本
    interpretation_text = "\n".join([
        f"{k.replace('_', ' ').title()}: {invariants_interpretation.get(k, '')}"
        for k in ['spectral_gap', 'girth', 'chromatic_number']
    ])
    
    invariants_list = {
        'Spectral Gap': invariants.get('spectral_gap', np.nan),
        'Girth': invariants.get('girth', np.nan),
        'Chromatic Number': invariants.get('chromatic_number', np.nan),
    }
    
    valid_invariants = {k: v for k, v in invariants_list.items() if not np.isnan(v)}
    
    if valid_invariants:
        plt.bar(valid_invariants.keys(), valid_invariants.values(), color='skyblue')
        plt.title('Graph Invariants with Immunological Relevance')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        
        for i, v in enumerate(valid_invariants.values()):
            plt.text(i, v + 0.05, f'{v:.4f}', ha='center', fontsize=9)
        
        # 添加解释文本
        plt.figtext(0.5, 0.01, interpretation_text, wrap=True, ha="center", fontsize=9)
        
        invariants_path = os.path.join(output_dir, f'graph_invariants_{method_name}.png')
        plt.tight_layout()
        plt.savefig(invariants_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Created graph invariants visualization: {invariants_path}")
    
    # 4. UMAP可视化（包含免疫学特性）
    if len(filtered_features) > 0 and len(filtered_features[0]) >= 4:
        plt.figure(figsize=(10, 8))
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(np.array(filtered_features)[:, :4])
        
        # 添加免疫学特征作为额外维度
        antigen_labels = [filtered_G.nodes[n]['antigen_specificity'] for n in filtered_G.nodes]
        unique_antigens = sorted(set(antigen_labels))
        
        # 为每个抗原类型绘图
        for antigen in unique_antigens:
            idx = [i for i, a in enumerate(antigen_labels) if a == antigen]
            if len(idx) > 0:
                plt.scatter(
                    embeddings[idx, 0], embeddings[idx, 1],
                    label=antigen,
                    s=np.array(filtered_features)[idx, 0]*50 + 10,  # 点大小基于频率
                    alpha=0.7
                )
        
        plt.title(f'UMAP Projection with Antigen Specificity ({method_name})')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        # 添加免疫学解释
        dominant_antigen = max(Counter(antigen_labels), key=Counter(antigen_labels).get)
        plt.figtext(0.5, 0.01, f"Dominant antigen specificity: {dominant_antigen}", 
                   ha="center", fontsize=9)
        
        if len(unique_antigens) <= 15:
            plt.legend(title='Antigen Specificity', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        umap_path = os.path.join(output_dir, f'umap_antigen_{method_name}.png')
        plt.tight_layout()
        plt.savefig(umap_path, dpi=300)
        plt.close()
        logging.info(f"Created UMAP visualization: {umap_path}")
    
    # 5. 特征分布图（pairplot）- 增强免疫学标注
    if len(filtered_features) > 0:
        # 准备特征数据
        feature_names = ['frequency', 'cdr3_length', 'centrality', 'community']
        if len(filtered_features[0]) > 4:
            feature_names.append('eigen_centrality')
            
        feature_df = pd.DataFrame(
            filtered_features,
            columns=feature_names
        )
        
        # 添加抗原标签
        feature_df['antigen_specificity'] = [filtered_G.nodes[n]['antigen_specificity'] for n in filtered_G.nodes]
        
        # 当抗原类型过多时，只显示主要的前10种
        antigen_counts = feature_df['antigen_specificity'].value_counts()
        if len(antigen_counts) > 15:
            top_antigens = antigen_counts.index[:10].tolist()
            feature_df['antigen_specificity'] = feature_df['antigen_specificity'].apply(
                lambda x: x if x in top_antigens else 'OTHER')
        
        try:
            # 创建基本图形
            plt.figure(figsize=(12, 8))
            
            # 使用 pairplot
            g = sns.pairplot(
                feature_df, 
                hue='antigen_specificity', 
                palette='tab10',
                diag_kind='kde',
                plot_kws={'alpha': 0.6}
            )
            
            # 添加整体标题和免疫学解释
            plt.suptitle(f'Feature Distributions ({method_name})', y=1.02)
            plt.figtext(0.5, -0.05, "Feature distributions show relationships between sequence properties and antigen specificity", 
                       ha="center", fontsize=10)
            
            # 保存图像
            dist_path = os.path.join(output_dir, f'feature_dist_{method_name}.png')
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Created feature distribution visualization: {dist_path}")
        except Exception as e:
            logging.error(f"Error creating feature distribution plot: {str(e)}")
    
    # 6. 抗原特异性分布图 - 添加免疫学解释
    if len(filtered_features) > 0:
        try:
            antigen_counts = feature_df['antigen_specificity'].value_counts().reset_index()
            antigen_counts.columns = ['antigen_specificity', 'count']
            
            # 如果类型太多，聚合较小的类型
            if len(antigen_counts) > 15:
                main_antigens = antigen_counts.head(10)
                other_count = antigen_counts[10:]['count'].sum()
                antigen_counts = pd.concat([
                    main_antigens,
                    pd.DataFrame([['OTHER', other_count]], columns=['antigen_specificity', 'count'])
                ])
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='antigen_specificity', y='count', data=antigen_counts)
            plt.title(f'Distribution of Antigen Specificities ({method_name})')
            plt.xlabel('Antigen Specificity')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # 添加免疫学解释
            dominant_antigen = antigen_counts.iloc[0]['antigen_specificity']
            dominant_count = antigen_counts.iloc[0]['count']
            total = antigen_counts['count'].sum()
            plt.figtext(0.5, -0.1, f"Dominant specificity: {dominant_antigen} ({dominant_count/total:.1%})", 
                       ha="center", fontsize=10)
            
            antigen_path = os.path.join(output_dir, f'antigen_distribution_{method_name}.png')
            plt.tight_layout()
            plt.savefig(antigen_path, dpi=300)
            plt.close()
            logging.info(f"Created antigen distribution visualization: {antigen_path}")
        except Exception as e:
            logging.error(f"Error creating antigen distribution plot: {str(e)}")

# 6. 表格输出 - 增强免疫学信息
def generate_tables(G, features, invariants, immunological_metrics, invariants_interpretation, output_dir, method_name):
    """生成分析结果表格 - 增加免疫学指标"""
    logging.info("Generating result tables with immunological insights...")
    os.makedirs(output_dir, exist_ok=True)
    
    tables_created = []
    
    # 节点特征表
    node_data = []
    for node, data in G.nodes(data=True):
        node_data.append({
            'sequence': node,
            'frequency': data['frequency'],
            'v_gene': data.get('v_gene', 'N/A'),
            'j_gene': data.get('j_gene', 'N/A'),
            'cdr3_length': data.get('cdr3_length', 0),
            'centrality': data.get('centrality', 0),
            'community': data.get('community', -1),
            'antigen_specificity': data.get('antigen_specificity', 'NAN'),
            'degree': G.degree[node],
            'dominant_antigen_in_community': data.get('dominant_antigen', 'N/A')
        })
    
    node_df = pd.DataFrame(node_data)
    node_table_path = os.path.join(output_dir, f'node_features_{method_name}.csv')
    node_df.to_csv(node_table_path, index=False)
    tables_created.append(node_table_path)
    logging.info(f"Created node feature table: {node_table_path}")
    
    # 边特征表 - 增加免疫学相关信息
    edge_data = []
    for u, v, data in G.edges(data=True):
        u_antigen = G.nodes[u]['antigen_specificity']
        v_antigen = G.nodes[v]['antigen_specificity']
        same_antigen = 'yes' if u_antigen == v_antigen and u_antigen != 'NAN' else 'no'
        edge_data.append({
            'source': u,
            'target': v,
            'weight': data.get('weight', 0),
            'source_antigen': u_antigen,
            'target_antigen': v_antigen,
            'same_antigen': same_antigen
        })
    
    if edge_data:  # 只有在有边的情况下才创建表
        edge_df = pd.DataFrame(edge_data)
        edge_table_path = os.path.join(output_dir, f'edge_features_{method_name}.csv')
        edge_df.to_csv(edge_table_path, index=False)
        tables_created.append(edge_table_path)
        logging.info(f"Created edge feature table: {edge_table_path}")
    
    # 网络整体指标表 - 添加免疫学解释
    if len(G.nodes) > 0:
        metrics = {
            'Similarity Method': method_name,
            'Total Nodes': len(G.nodes),
            'Total Edges': len(G.edges),
            'Average Degree': sum(d for n, d in G.degree()) / len(G.nodes),
            'Connected Components': nx.number_connected_components(G),
            'Average Clustering': nx.average_clustering(G),
            'Spectral Gap': invariants.get('spectral_gap', np.nan),
            'Immunological Interpretation': invariants_interpretation.get('spectral_gap', ''),
            'Girth': invariants.get('girth', np.nan),
            'Chromatic Number': invariants.get('chromatic_number', np.nan),
            'Chromatic Interpretation': invariants_interpretation.get('chromatic_number', ''),
            'Network Diameter': invariants.get('diameter', 'Disconnected' if invariants.get('diameter', np.nan) is np.nan else invariants['diameter']),
            'Average Path Length': invariants.get('average_path_length', 'Disconnected' if invariants.get('average_path_length', np.nan) is np.nan else invariants['average_path_length']),
            'GPU Acceleration': 'Yes' if USE_GPU else 'No',
            'Processing Time': time.time() - start_time
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_table_path = os.path.join(output_dir, f'network_metrics_{method_name}.csv')
        metrics_df.to_csv(metrics_table_path, index=False)
        tables_created.append(metrics_table_path)
        logging.info(f"Created network metrics table: {metrics_table_path}")
    
    # 抗原特异性分析表 - 增强免疫学信息
    if node_data:
        antigen_stats = pd.DataFrame(node_data).groupby('antigen_specificity').agg(
            sequence_count=('sequence', 'count'),
            mean_frequency=('frequency', 'mean'),
            median_frequency=('frequency', 'median'),
            min_frequency=('frequency', 'min'),
            max_frequency=('frequency', 'max'),
            mean_cdr3_length=('cdr3_length', 'mean'),
            median_cdr3_length=('cdr3_length', 'median'),
            mean_centrality=('centrality', 'mean')
        ).reset_index()
        
        antigen_table_path = os.path.join(output_dir, f'antigen_statistics_{method_name}.csv')
        antigen_stats.to_csv(antigen_table_path, index=False)
        tables_created.append(antigen_table_path)
        logging.info(f"Created antigen statistics table: {antigen_table_path}")
    
    # 新增：免疫学指标表
    immuno_table_data = {
        'Metric': ['Shannon Diversity Index', 'Cross-Reactivity Index', 'Mean CDR3 Length'],
        'Value': [
            immunological_metrics.get('shannon_diversity', np.nan),
            immunological_metrics.get('cross_reactivity_index', np.nan),
            immunological_metrics.get('cdr3_length_distribution', {}).get('mean', np.nan)
        ],
        'Interpretation': [
            immunological_metrics.get('interpretation', {}).get('shannon_diversity', ''),
            immunological_metrics.get('interpretation', {}).get('cross_reactivity', ''),
            immunological_metrics.get('interpretation', {}).get('cdr3_length', '')
        ]
    }
    
    immuno_df = pd.DataFrame(immuno_table_data)
    immuno_table_path = os.path.join(output_dir, f'immunological_metrics_{method_name}.csv')
    immuno_df.to_csv(immuno_table_path, index=False)
    tables_created.append(immuno_table_path)
    logging.info(f"Created immunological metrics table: {immuno_table_path}")
    
    # 社区分析表 - 添加免疫学信息
    if 'community' in node_df.columns:
        community_stats = node_df.groupby('community').agg(
            node_count=('sequence', 'count'),
            mean_frequency=('frequency', 'mean'),
            max_frequency=('frequency', 'max'),
            mean_centrality=('centrality', 'mean'),
            antigen_diversity=('antigen_specificity', pd.Series.nunique),
            dominant_antigen=('dominant_antigen_in_community', lambda x: x.mode()[0]),
            antigen_purity=('antigen_specificity', lambda x: x.value_counts(normalize=True).iloc[0])
        ).reset_index()
        community_table_path = os.path.join(output_dir, f'community_statistics_{method_name}.csv')
        community_stats.to_csv(community_table_path, index=False)
        tables_created.append(community_table_path)
        logging.info(f"Created community statistics table: {community_table_path}")
    
    # 算法公平性分析表 - 添加免疫学解释
    try:
        fairness_df = analyze_algorithmic_fairness(G)
        if not fairness_df.empty:
            fairness_table_path = os.path.join(output_dir, f'fairness_analysis_{method_name}.csv')
            fairness_df.to_csv(fairness_table_path, index=False)
            tables_created.append(fairness_table_path)
            logging.info(f"Created fairness analysis table: {fairness_table_path}")
    except Exception as e:
        logging.error(f"Fairness analysis failed: {str(e)}")
    
    # 信息传播模拟结果表 - 包含免疫学解释
    try:
        if len(G.nodes) > 0:
            # 选择中心性最高的节点作为种子
            centralities = nx.degree_centrality(G)
            seed_nodes = sorted(centralities, key=centralities.get, reverse=True)[:5]
            
            infected, interpretation = simulate_information_diffusion(G, seed_nodes)
            
            # 保存感染节点
            diffusion_df = pd.DataFrame({
                'seed_nodes': [",".join(seed_nodes)],
                'infected_count': [len(infected)],
                'infected_fraction': [len(infected) / len(G.nodes)],
                'immunological_interpretation': [interpretation]
            })
            
            # 创建节点状态表
            node_status = []
            for node in G.nodes:
                node_status.append({
                    'node': node,
                    'infected': 1 if node in infected else 0,
                    'antigen_specificity': G.nodes[node].get('antigen_specificity', 'NAN')
                })
            
            node_status_df = pd.DataFrame(node_status)
            
            diffusion_table_path = os.path.join(output_dir, f'diffusion_results_{method_name}.csv')
            node_status_table_path = os.path.join(output_dir, f'node_infection_status_{method_name}.csv')
            
            diffusion_df.to_csv(diffusion_table_path, index=False)
            node_status_df.to_csv(node_status_table_path, index=False)
            
            tables_created.extend([diffusion_table_path, node_status_table_path])
            logging.info(f"Created diffusion simulation tables")
    except Exception as e:
        logging.error(f"Diffusion simulation failed: {str(e)}")
    
    return tables_created

# 比较实验 - 添加免疫学指标
def run_comparative_experiment(df, output_dir):
    """运行比较实验并生成结果表格 - 增加免疫学指标"""
    logging.info("Running comparative experiments with immunological metrics...")
    
    # 声明全局变量
    global SIMILARITY_METHOD, USE_GPU
    
    # 1. 不同相似性阈值对网络连通性的影响 - 添加免疫学指标
    thresholds = [0.5, 0.6, 0.7, 0.8]
    threshold_results = []
    
    for threshold in thresholds:
        try:
            start_time_exp = time.time()
            G = build_metanet(df, similarity_threshold=threshold)
            features, G, invariants, immunological_metrics, _ = analyze_network(G)
            exp_time = time.time() - start_time_exp
            
            threshold_results.append({
                'Threshold': threshold,
                'Nodes': G.number_of_nodes(),
                'Edges': G.number_of_edges(),
                'Components': nx.number_connected_components(G),
                'Largest Component': len(max(nx.connected_components(G), key=len)) if G.nodes else 0,
                'Clustering': nx.average_clustering(G),
                'Diameter': invariants.get('diameter', np.nan),
                'Spectral Gap': invariants.get('spectral_gap', np.nan),
                'Diversity Index': immunological_metrics.get('shannon_diversity', np.nan),
                'Cross-Reactivity': immunological_metrics.get('cross_reactivity_index', np.nan),
                'Processing Time (s)': exp_time
            })
        except Exception as e:
            logging.error(f"Threshold experiment failed for {threshold}: {str(e)}")
            threshold_results.append({
                'Threshold': threshold,
                'Error': str(e)
            })
    
    # 保存结果
    threshold_df = pd.DataFrame(threshold_results)
    threshold_path = os.path.join(output_dir, 'threshold_experiment_results.csv')
    threshold_df.to_csv(threshold_path, index=False)
    logging.info(f"Created threshold experiment table: {threshold_path}")
    
    # 2. GPU vs CPU性能对比
    # 只对中等大小数据集运行，以避免过长时间
    sample_size = min(3000, len(df)) if not df.empty else 0
    if sample_size > 100:
        devices = ['GPU', 'CPU'] if torch.cuda.is_available() else ['CPU']
        device_results = []
        
        for device in devices:
            try:
                use_gpu_flag = (device == 'GPU')
                
                start_time_exp = time.time()
                # 创建小的测试集
                test_df = df.sample(sample_size)
                
                # 临时设置全局变量
                original_gpu_setting = USE_GPU
                USE_GPU = use_gpu_flag
                
                # 构建网络
                G = build_metanet(test_df)
                
                # 恢复设置
                USE_GPU = original_gpu_setting
                
                exp_time = time.time() - start_time_exp
                
                device_results.append({
                    'Device': device,
                    'Nodes': G.number_of_nodes(),
                    'Edges': G.number_of_edges(),
                    'Processing Time (s)': exp_time,
                    'Time per Node (ms)': exp_time / len(G.nodes) * 1000 if G.nodes else np.nan
                })
            except Exception as e:
                logging.error(f"Device experiment failed for {device}: {str(e)}")
                device_results.append({
                    'Device': device,
                    'Error': str(e)
                })
        
        device_df = pd.DataFrame(device_results)
        device_path = os.path.join(output_dir, 'device_comparison_results.csv')
        device_df.to_csv(device_path, index=False)
        logging.info(f"Created device comparison table: {device_path}")
    
    # 3. 不同数据规模的扩展性测试
    sample_sizes = [1000, 5000, 10000]
    scalability_results = []
    
    for size in sample_sizes:
        try:
            if len(df) > size:
                test_df = df.sample(size)
                
                start_time_exp = time.time()
                G = build_metanet(test_df)
                exp_time = time.time() - start_time_exp
                
                scalability_results.append({
                    'Sample Size': size,
                    'Processing Time (s)': exp_time,
                    'Nodes': len(G.nodes),
                    'Edges': len(G.edges),
                    'Time per Node (ms)': exp_time / len(G.nodes) * 1000 if G.nodes else np.nan
                })
        except Exception as e:
            logging.error(f"Scalability experiment failed for size {size}: {str(e)}")
            scalability_results.append({
                'Sample Size': size,
                'Error': str(e)
            })
    
    scalability_df = pd.DataFrame(scalability_results)
    scalability_path = os.path.join(output_dir, 'scalability_results.csv')
    scalability_df.to_csv(scalability_path, index=False)
    logging.info(f"Created scalability test table: {scalability_path}")
    
    # 4. 算法性能对比（ImmunoBERT vs Levenshtein）- 添加免疫学指标
    method_results = []
    methods = ['levenshtein', 'immunobert'] if 'immunobert' in SIMILARITY_METHOD else [SIMILARITY_METHOD]
    
    for method in methods:
        try:
            start_time_exp = time.time()
            # 创建小的测试集
            test_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
            
            # 临时设置全局变量
            original_method = SIMILARITY_METHOD
            SIMILARITY_METHOD = method
            
            # 构建网络
            G = build_metanet(test_df)
            features, G, invariants, immunological_metrics, _ = analyze_network(G)
            
            # 恢复设置
            SIMILARITY_METHOD = original_method
            
            exp_time = time.time() - start_time_exp
            
            method_results.append({
                'Method': method,
                'Processing Time (s)': exp_time,
                'Similarity Threshold': SIMILARITY_THRESHOLD,
                'Network Nodes': len(G.nodes),
                'Network Edges': len(G.edges),
                'Average Clustering': nx.average_clustering(G),
                'Spectral Gap': invariants.get('spectral_gap', np.nan),
                'Diversity Index': immunological_metrics.get('shannon_diversity', np.nan),
                'Cross-Reactivity': immunological_metrics.get('cross_reactivity_index', np.nan),
                'Time per Node (ms)': exp_time / len(G.nodes) * 1000 if G.nodes else np.nan
            })
        except Exception as e:
            logging.error(f"Method comparison failed for {method}: {str(e)}")
            method_results.append({
                'Method': method,
                'Error': str(e)
            })
    
    method_df = pd.DataFrame(method_results)
    method_path = os.path.join(output_dir, 'method_comparison_results.csv')
    method_df.to_csv(method_path, index=False)
    logging.info(f"Created method comparison table: {method_path}")
    
    return {
        'threshold': threshold_path,
        'device': device_path if 'device_df' in locals() else None,
        'scalability': scalability_path,
        'method': method_path
    }

# 主函数 - 添加进度跟踪和免疫学输出
# 主函数 - 添加进度跟踪和免疫学输出
def process_bcr_tcr_data(input_dir, output_base_dir):
    """处理目录中的所有BCR/TCR文件 - 增强免疫学输出"""
    # 声明全局变量
    global SIMILARITY_METHOD, USE_GPU, start_time
    
    start_time = time.time()
    processed_files = 0
    os.makedirs(output_base_dir, exist_ok=True)
    
    logging.info("="*60)
    logging.info("Starting BCR/TCR Network Analysis Pipeline with Immunological Focus")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_base_dir}")
    logging.info(f"Similarity method: {SIMILARITY_METHOD}")
    logging.info(f"GPU acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    logging.info(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    logging.info("="*60)
    
    # 获取所有文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.uniq.fre.sort.anno.CDR3.com')]
    
    if not files:
        logging.warning("No compatible files found in input directory")
        return
    
    # 比较两种方法
    methods = ['levenshtein', 'immunobert'] if SIMILARITY_METHOD == 'compare' else [SIMILARITY_METHOD]
    
    for file_name in files:
        sample_id = file_name.split('.')[0]
        
        for method in methods:
            method_output_dir = os.path.join(output_base_dir, sample_id, method)
            os.makedirs(method_output_dir, exist_ok=True)
            
            # 设置每个样本的单独日志文件
            log_file = os.path.join(method_output_dir, 'processing_log.txt')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            
            logger = logging.getLogger()
            logger.addHandler(file_handler)
            
            try:
                logging.info(f"Processing file: {file_name} with method: {method}")
                logging.info(f"Immunological analysis enabled for this sample")
                file_path = os.path.join(input_dir, file_name)
                
                # 1. 加载数据
                df = load_and_preprocess(file_path)
                
                # 运行比较实验（仅在第一个方法上运行）
                if method == methods[0] and len(df) >= 1000:
                    run_comparative_experiment(df, method_output_dir)
                
                # 检查数据是否足够
                if df.empty or len(df) < 2:
                    logging.warning(f"Skipping {file_name}: insufficient data after filtering")
                    continue
                
                # 保存原始方法设置
                original_method = SIMILARITY_METHOD
                
                # 临时设置方法
                SIMILARITY_METHOD = method
                
                # 2. 构建网络
                G = build_metanet(df)
                
                # 检查网络是否有效
                if G.number_of_nodes() == 0:
                    logging.warning(f"Skipping {file_name}: no valid network created")
                    continue
                
                # 3. 分析网络（含免疫学指标）
                features, G, invariants, immunological_metrics, invariants_interpretation = analyze_network(G)
                
                # 4. 可视化（增强免疫学输出）
                visualize_results(G, features, invariants, immunological_metrics, invariants_interpretation, method_output_dir, method)
                
                # 5. 生成表格（增强免疫学输出）
                generate_tables(G, features, invariants, immunological_metrics, invariants_interpretation, method_output_dir, method)
                
                processed_files += 1
                logging.info(f"Successfully processed {file_name} with {method} - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                
                # 输出免疫学摘要
                antigen_counts = Counter(nx.get_node_attributes(G, 'antigen_specificity').values())
                dominant_antigen = max(antigen_counts, key=antigen_counts.get) if antigen_counts else "N/A"
                diversity = immunological_metrics.get('shannon_diversity', 0)
                cross_reactivity = immunological_metrics.get('cross_reactivity_index', 0)
                logging.info(f"Immunological summary: Dominant antigen = {dominant_antigen}, " \
                            f"Diversity = {diversity:.3f}, Cross-reactivity index = {cross_reactivity:.3f}")
                
            except Exception as e:
                logging.error(f"Error processing {file_name} with {method}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 保存错误信息
                error_file = os.path.join(method_output_dir, 'ERROR.txt')
                with open(error_file, 'w') as f:
                    f.write(f"Error processing {file_path} with method {method}:\n")
                    f.write(str(e))
                    f.write("\n\nTraceback:\n")
                    traceback.print_exc(file=f)
            
            finally:
                # 确保恢复原始方法（即使在出错的情况下）
                SIMILARITY_METHOD = original_method
                
                # 移除文件日志处理器
                logger.removeHandler(file_handler)
                file_handler.close()
    
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info(f"Processing complete! Processed {processed_files}/{len(files)} files in {total_time:.2f} seconds")
    logging.info(f"Results saved to: {output_base_dir}")
    logging.info("="*60)

# 运行主程序
if __name__ == "__main__":
    input_directory = "/home/ubuntu-user/624/624-metanet-main/20250402"
    output_directory = "/home/ubuntu-user/624/624-metanet-main/results05"
    
    # 配置选项（这些设置会覆盖文件开头的默认设置）
    SIMILARITY_METHOD = 'compare'  # 'levenshtein', 'immunobert' 或 'compare'
    USE_GPU = True  # 是否使用GPU加速
    
    process_bcr_tcr_data(input_directory, output_directory)