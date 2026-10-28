import csv
import torch
import numpy as np
import os
import re
from tqdm import tqdm
from Bio import pairwise2
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
from transformers import T5EncoderModel, T5Tokenizer

# ======================== 配置路径（直接修改此处即可） ========================
CSV_PATH = "/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/data/ABP/abp_data/Se.csv"          # 输入的多肽CSV文件路径（无表头，每行一个序列）
PDB_DIR = "/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/data/ABP/esmfold/Se"               # PDB文件存放目录（文件名：0.pdb, 1.pdb...）
PROTTRANS_OUT = "/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/data/ABP/prottrans_output/Se"       # ProtTrans嵌入输出目录
PDB_OUT = "/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/data/ABP/pdb_coords/Se"            # PDB坐标输出目录
DSSP_OUT = "/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/data/ABP/dssp_out/Se"        # DSSP特征输出目录
DSSP_BIN = "/user/data1/lijinyu/miniconda3/envs/ljy_310/bin/mkdssp"         # DSSP可执行文件路径（根据实际安装位置修改）
# ==========================================================================


warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def sanitize_filename(seq):
    """生成安全的文件名（保留字母、数字和下划线）"""
    return re.sub(r'[^A-Za-z0-9_]', '', seq.replace(" ", "_"))

def get_prottrans():
    """生成ProtTrans嵌入（按序列名命名）"""
    os.makedirs(PROTTRANS_OUT, exist_ok=True)
    
    # 读取CSV并获取原始序列
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        raw_seqs = [row[0] for row in reader if row]
    
    # 加载模型
    tokenizer = T5Tokenizer.from_pretrained("/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/Prot-T5-XL-U50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/Prot-T5-XL-U50").eval().cuda()
    
    # 处理每个序列
    for raw_seq in tqdm(raw_seqs, desc="生成ProtTrans嵌入"):
        # 1. 严格过滤无效字符
        seq = re.sub(r"[^A-Z]", "", raw_seq.upper()).strip()
        if not seq:
            continue
            
        # 2. 生成token（不自动添加特殊标记）
        inputs = tokenizer(
            " ".join(seq),
            return_tensors="pt",
            add_special_tokens=False  # 关键修改
        ).to(model.device)
        
        # 3. 验证输入长度
        if inputs.input_ids.shape[1] != len(seq):
            print(f"Token数量异常: {seq} → {inputs.input_ids.shape[1]}")
            continue
            
        # 4. 生成嵌入
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[0].cpu().numpy()
        
        # 5. 强制校验
        if emb.shape[0] != len(seq):
            print(f"特征维度异常: {seq} → {emb.shape}")
            continue
            
        # 保存文件
        seq_name = sanitize_filename(seq)
        np.save(os.path.join(PROTTRANS_OUT, f"{seq_name}_prottrans.npy"), emb)

def proc_pdb():
    """处理PDB坐标（按序列名命名）"""
    os.makedirs(PDB_OUT, exist_ok=True)
    
    # 读取原始序列
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        raw_seqs = [row[0] for row in reader if row]
    
    # 处理每个PDB
    for raw_seq in tqdm(raw_seqs, desc="处理PDB坐标"):
        seq = raw_seq.replace(" ", "")
        seq_name = sanitize_filename(raw_seq)
        pdb_path = os.path.join(PDB_DIR, f"{seq_name}.pdb")
        
        if not os.path.exists(pdb_path):
            print(f"\nPDB文件不存在: {seq_name}")
            continue
            
        # 提取坐标
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        coord = _get_pdb_xyz(lines, seq)  # 传入原始序列进行校验
        
        if coord is not None:
            np.save(os.path.join(PDB_OUT, f"{seq_name}_coordinates.npy"), coord)
        else:
            print(f"\n坐标校验失败: {seq_name}")

def _get_pdb_xyz(pdb_lines, ref_seq):
    """精确的CA坐标提取与残基校验"""
    ca_coords = []          # 存储CA坐标
    max_res_num = 0         # 记录最大残基序号
    
    for line in pdb_lines:
        # 过滤非ATOM记录
        if not line.startswith("ATOM"):
            continue
        
        # 严格按列位置提取关键信息
        atom_type = line[12:16].strip()   # 13-16列：原子类型（CA）
        res_num_str = line[22:26].strip() # 23-26列：残基序号
        
        # 仅处理CA原子
        if atom_type != 'CA':
            continue
            
        # 提取坐标（31-54列）
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            print(f"坐标解析失败：{line.strip()}")
            continue
        
        # 转换残基序号为整数
        try:
            res_num = int(res_num_str)
        except ValueError:
            print(f"无效残基序号：{res_num_str}")
            continue
            
        # 更新最大残基序号
        if res_num > max_res_num:
            max_res_num = res_num
        
        ca_coords.append([x, y, z])
    
    # 关键校验逻辑
    if len(ca_coords) != max_res_num:
        print(f"校验失败：CA原子数({len(ca_coords)})≠最大残基序号({max_res_num})")
        print(f"示例验证：第{max_res_num}号残基应为最后一个CA原子")
        return None
    
    return np.array(ca_coords)

def get_dssp():
    """生成DSSP特征（按序列名命名）"""
    os.makedirs(DSSP_OUT, exist_ok=True)
    
    # 读取原始序列
    with open(CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        raw_seqs = [row[0] for row in reader if row]
    
    # 处理每个序列
    for raw_seq in tqdm(raw_seqs, desc="生成DSSP特征"):
        seq = raw_seq.replace(" ", "")
        seq_name = sanitize_filename(raw_seq)
        pdb_path = os.path.join(PDB_DIR, f"{seq_name}.pdb")
        
        if not os.path.exists(pdb_path):
            print(f"\nPDB文件不存在: {seq_name}")
            continue
            
        # 运行DSSP
        dssp_file = os.path.join(DSSP_OUT, f"{seq_name}.dssp")
        os.system(f"{DSSP_BIN} -i {pdb_path} -o {dssp_file} 2>/dev/null")
        
        # 解析特征
        try:
            dssp_seq, dssp_feat = _parse_dssp(dssp_file)
            if dssp_seq != seq:
                dssp_feat = _align_dssp(dssp_seq, dssp_feat, seq)
            np.save(os.path.join(DSSP_OUT, f"{seq_name}_dssp.npy"), _transform_dssp(dssp_feat))
            os.remove(dssp_file)
        except Exception as e:
            print(f"DSSP处理失败: {seq_name}, 错误: {str(e)}")

def _parse_dssp(dssp_file):
    """改进的DSSP解析"""
    try:
        with open(dssp_file, 'r') as f:
            lines = f.readlines()
        
        # 定位数据起始行
        start_line = next(i for i, line in enumerate(lines) if line.startswith('  #  RESIDUE'))
        lines = lines[start_line+1:]
        
        seq, feats = [], []
        for line in lines:
            if len(line) < 115:
                continue
                
            # 提取氨基酸和特征
            aa = line[13] if line[13] != '!' else 'X'
            ss = line[16] if line[16] in 'HBEGITSC' else 'C'
            
            # 处理缺失值
            try:
                phi = float(line[103:109].strip() or 360.0)
                psi = float(line[109:115].strip() or 360.0)
                acc = float(line[34:38].strip() or 0.0)
            except:
                phi = psi = 360.0
                acc = 0.0
                
            # 构建特征向量
            ss_vec = np.zeros(8)
            if ss in 'HBEGITSC':
                ss_vec['HBEGITSC'.index(ss)] = 1.0
                
            feat = np.concatenate([
                np.array([phi, psi, acc], dtype=np.float32),
                ss_vec
            ])
            
            seq.append(aa)
            feats.append(feat)
            
        return ''.join(seq), np.array(feats)
    except Exception as e:
        raise RuntimeError(f"DSSP解析失败: {str(e)}")

def _align_dssp(dssp_seq, dssp_feat, ref_seq):
    """修复的特征对齐"""
    align = pairwise2.align.globalxx(ref_seq, dssp_seq)[0]
    feat_padded = []
    ptr = 0
    
    for a_ref, a_dssp in zip(align.seqA, align.seqB):
        if a_ref == '-':
            continue
            
        if a_dssp == '-':
            # 正确填充13维特征 [phi, psi, acc, ss(8)]
            feat_padded.append(np.array([360.0, 360.0, 0.0] + [0.0]*8, dtype=np.float32))
        else:
            feat_padded.append(dssp_feat[ptr])
            ptr += 1
    
    return np.stack(feat_padded)

def _transform_dssp(feats):
    """修复的特征转换"""
    angles = np.deg2rad(feats[:, :2])
    sin_phi, cos_phi = np.sin(angles[:, 0]), np.cos(angles[:, 0])
    sin_psi, cos_psi = np.sin(angles[:, 1]), np.cos(angles[:, 1])
    return np.hstack([
        sin_phi[:, None], cos_phi[:, None],
        sin_psi[:, None], cos_psi[:, None],
        feats[:, 2:]
    ])

if __name__ == "__main__":
    get_prottrans()
    #proc_pdb()
    get_dssp()