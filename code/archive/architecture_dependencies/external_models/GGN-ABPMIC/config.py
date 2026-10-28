import torch
import os

class Config:
    def __init__(self):
        self.fp16 = True
        # ================ 数据路径 ================
        self.train_csv = os.environ.get("GGN_TRAIN_CSV", "../ggn_work/train.csv")  # [v2-9 patch]
        #self.val_csv = "/user/data1/lijinyu/snap/snapd-desktop-integration/GGN-GO-main/data/ABP/abp_data/St/test/test.csv"
        self.pdb_dir = os.environ.get("GGN_PDB_DIR", "../ggn_work/pdb")  # [v2-9 patch]
        self.prottrans_dir = os.environ.get("GGN_PROTTRANS_DIR", "../ggn_work/prottrans")  # [v2-9 patch]
        self.dssp_dir = os.environ.get("GGN_DSSP_DIR", "../ggn_work/dssp")  # [v2-9 patch]
        
        # ================ 训练参数 ================
        self.lr = 1e-4
        self.epochs = 100
        self.contrast = True
        self.contrast_weight = 0.3
        self.temperature = 0.2
        self.margin = 0.5
        self.drop_rate = 0.5
        self.num_workers = 0
        
        # ================ 系统参数 ================
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # [v2-9 patch]
        self.wandb = False  # [v2-9 patch]
        self.batch_size = 1000  # [v2-9 patch] BatchSampler node budget
        self.fp = self.fp16  # [v2-9 patch] name alias used by train.py
        self.save_dir = os.environ.get("GGN_SAVE", "../ggn_work/ckpt_default")  # [v2-9 patch]
        os.makedirs(self.save_dir, exist_ok=True)

# 创建全局配置实例
config = Config()