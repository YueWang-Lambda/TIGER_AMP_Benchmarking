import torch
torch.cuda.empty_cache()
torch.multiprocessing.set_start_method('spawn', force=True)
import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split  # 修改1: 导入train_test_split
from model import AMPRegressor
from nt_xent import MICContrastiveLoss
from myData import ProteinGraphDataset, BatchSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
import argparse
import warnings
import os
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import csv
from datetime import datetime

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

def collate_fn(batch):
    """智能过滤无效数据并生成批次"""
    valid_data = []
    for data in batch:
        if data is None or not hasattr(data, 'y'):
            continue
        required_attrs = ['x', 'node_s', 'edge_index', 'y','peptide_feat']
        if all(hasattr(data, attr) for attr in required_attrs):
            if data.edge_index.shape[1] > 0 and data.node_s.shape[0] == len(data.seq):
                valid_data.append(data)
    
    if not valid_data:
        dummy = Data(
            x=torch.zeros(1, 3),
            seq=torch.zeros(1, dtype=torch.long),
            node_s=torch.zeros(1, 1043),
            node_v=torch.zeros(1, 3, 3),
            edge_index=torch.zeros(2, 1, dtype=torch.long),
            edge_s=torch.zeros(1, 32),
            edge_v=torch.zeros(1, 1, 3),
            y=torch.tensor([0.0]),
            peptide_feat=torch.zeros(1, 553),  # 添加虚拟肽特征，确保是2D
            name="dummy"
        )
        valid_data.append(dummy)
        print("警告: 生成虚拟批次数据")
    
    return Batch.from_data_list(valid_data)

class SafeSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]  
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
        
    def __len__(self):
        return len(self.indices)

class AdaptiveHybridLoss(nn.Module):
    def __init__(self, num_samples, base_mse_weight=0.6, base_huber_weight=0.3, device='cuda'):
        super().__init__()
        self.num_samples = num_samples
        self.base_mse_weight = base_mse_weight
        self.base_huber_weight = base_huber_weight
        self.device = device
        
        # Delta预测网络
        self.delta_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        ).to(self.device)
        
        self.contrast_loss = MICContrastiveLoss().to(self.device)
        
    def to(self, device):
        super().to(device)
        self.device = device
        self.delta_net = self.delta_net.to(device)
        self.contrast_loss = self.contrast_loss.to(device)
        return self
        
    def adaptive_delta(self, targets):
        try:
            targets = targets.to(self.device)
            std = torch.std(targets).view(1, 1)
            delta = self.delta_net(std).clamp(min=0.1, max=2.0).item()
            return max(0.1, delta)
        except:
            return 1.0
        
    def forward(self, pred, emb, y_true, epoch, max_epoch):
        pred = pred.to(self.device)
        emb = emb.to(self.device)
        y_true = y_true.to(self.device)
        
        sample_factor = min(1.0, self.num_samples / 1000)
        mse_weight = self.base_mse_weight * sample_factor
        huber_weight = self.base_huber_weight
        contrast_weight = 0.5 * (1 - epoch/max_epoch)
        
        mse = F.smooth_l1_loss(pred, y_true, beta=0.1)
        delta = self.adaptive_delta(y_true)
        huber = F.huber_loss(pred, y_true, delta=delta)
        contrast = self.contrast_loss(emb, y_true)
        
        if epoch < max_epoch//3:
            return 0.8*mse + 0.2*huber
        elif epoch < 2*max_epoch//3:
            return (0.6*mse_weight*mse + 0.3*huber_weight*huber + 0.1*contrast_weight*contrast)
        else:
            return (0.3*mse_weight*mse + 0.3*huber_weight*huber + 0.4*contrast_weight*contrast)

def train_epoch(model, loader, optimizer, scaler, epoch, max_epoch, loss_fn):
    model.train()
    total_loss = 0
    torch.cuda.empty_cache()
    
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        if batch is None or batch.num_graphs == 0:
            continue
            
        optimizer.zero_grad()
        batch = batch.to(model.device)
        
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            pred, emb = model(batch)
            loss = loss_fn(pred, emb, batch.y.view(-1), epoch, max_epoch)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
        if batch.num_graphs < 100:  
            torch.cuda.empty_cache()
    
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn, epoch, max_epoch):
    model.eval()
    preds, truths, val_loss = [], [], 0
    with torch.no_grad():
        for batch in loader:
            if batch is None or batch.num_graphs == 0:
                continue
            batch = batch.to(model.device)
            p, emb = model(batch)
            preds.append(p.cpu())
            truths.append(batch.y.view(-1).cpu())
            val_loss += loss_fn(p, emb, batch.y.view(-1), epoch, max_epoch).item()
    
    if not preds or len(preds) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), None, None, float('nan')
    
    try:
        preds = torch.cat(preds).numpy()
        truths = torch.cat(truths).numpy()
        
        if np.var(truths) < 1e-6 or np.var(preds) < 1e-6:
            print("警告: 数据方差过小，指标无效")
            return float('inf'), 0.0, 0.0, float('inf'), None, None, float('inf')
        
        rmse = np.sqrt(mean_squared_error(truths, preds))
        mse = mean_squared_error(truths, preds)
        r2 = r2_score(truths, preds)
        pcc, _ = pearsonr(truths, preds)
        return rmse, r2, pcc, mse, preds, truths, val_loss/len(loader)
    except Exception as e:
        print(f"评估错误: {str(e)}")
        return float('inf'), 0.0, 0.0, float('inf'), None, None, float('inf')

def plot_results(true, pred, epoch, save_dir):
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.scatter(true, pred, alpha=0.3)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Epoch {epoch} Prediction')
    
    plt.subplot(122)
    errors = true - pred
    plt.hist(errors, bins=50)
    plt.title('Error Distribution')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.close()

def train_kfold(config):
    # 添加损失记录列表
    loss_history = []  # 存储每个epoch的损失历史
    
    # 初始化时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    torch.cuda.set_per_process_memory_fraction(0.9, device=torch.device(config.device).index)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"\n=== 使用设备: {device} ===")
    print(f"当前GPU{device.index}显存状态:")
    print(torch.cuda.memory_summary(device=device))
    
    if config.wandb:
        try:
            import wandb
            wandb.init(project="AMP-Regression", config=vars(config))
        except ImportError:
            config.wandb = False

    full_dataset = ProteinGraphDataset(config)

    # 验证肽特征维度
    sample = full_dataset[0]
    if not hasattr(sample, 'peptide_feat'):
        raise AttributeError("Data对象缺少peptide_feat属性")
    if sample.peptide_feat.dim() != 2 or sample.peptide_feat.shape[1] != 553:
        raise ValueError(f"肽特征维度错误，应为[N,553]，实际得到{sample.peptide_feat.shape}")

    if len(full_dataset) == 0:
        raise ValueError("数据集为空")
    
    if len(full_dataset) < 1000:
        config.batch_size = max(4, len(full_dataset)//10)
        config.num_workers = 0

    # 初始化日志文件
    log_file = os.path.join(config.save_dir, f"training_results_{timestamp}.csv")
    print(f"训练日志路径: {log_file}")

    # 创建保存划分序列的目录
    sequence_splits_dir = os.path.join(config.save_dir, "sequence_splits")
    os.makedirs(sequence_splits_dir, exist_ok=True)
    print(f"序列划分文件将保存在: {sequence_splits_dir}")

    best_results = {}
    # 将5折交叉验证改为10次随机划分
    n_splits = int(os.environ.get('GGN_SEEDS', '10'))  # [v2-9 patch]
    split_results = []
    
    all_indices = np.arange(len(full_dataset))
    
    # 设置随机数种子范围，确保每次运行结果可重现
    _base = int(os.environ.get('GGN_SEED_BASE', '42'))  # [v2-9 patch]
    random_seeds = [_base + i for i in range(n_splits)]
    
    for split_idx in range(n_splits):
        print(f"\n===== 随机划分 {split_idx+1}/{n_splits} =====")
        
        # 设置当前划分的随机种子
        seed = random_seeds[split_idx]
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 使用train_test_split进行80:20的划分
        train_idx, val_idx = train_test_split(
            all_indices, 
            test_size=0.2, 
            random_state=seed
        )
        
        # 保存训练集序列
        train_sequences = [full_dataset.sequences[i] for i in train_idx]
        train_csv_path = os.path.join(sequence_splits_dir, f"random_split{split_idx+1}_train_sequences.csv")
        with open(train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence'])
            for seq in train_sequences:
                writer.writerow([seq])
        print(f"保存训练集序列: {train_csv_path} ({len(train_sequences)}个序列)")
        
        # 保存验证集序列
        val_sequences = [full_dataset.sequences[i] for i in val_idx]
        val_csv_path = os.path.join(sequence_splits_dir, f"random_split{split_idx+1}_val_sequences.csv")
        with open(val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence'])
            for seq in val_sequences:
                writer.writerow([seq])
        print(f"保存验证集序列: {val_csv_path} ({len(val_sequences)}个序列)")

        best_rmse, best_r2, best_pcc = float('inf'), None, None
        
        train_set = SafeSubset(full_dataset, train_idx)
        val_set = SafeSubset(full_dataset, val_idx)
        
        train_loader = DataLoader(
            train_set,
            batch_sampler=BatchSampler(
                [len(full_dataset.sequences[i]) for i in train_idx],
                config.batch_size, shuffle=True
            ),
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )
        
        val_loader = DataLoader(
            val_set,
            batch_sampler=BatchSampler(
                [len(full_dataset.sequences[i]) for i in val_idx],
                config.batch_size, shuffle=False
            ),
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )

        model = AMPRegressor(
            input_dim=1043,
            hidden_dim=256,
            drop_rate=config.drop_rate,
            device=device
        ).to(device)
        
        loss_fn = AdaptiveHybridLoss(
            num_samples=len(train_set),
            device=config.device
        )
        optimizer = model.configure_optimizers(lr=config.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        
        best_rmse, patience = float('inf'), 30
        for epoch in range(config.epochs):
            try:
                train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch, config.epochs, loss_fn)
                rmse, r2, pcc, mse, val_pred, val_true, val_loss = evaluate(model, val_loader, loss_fn, epoch, config.epochs)
                
                if np.isnan(rmse) or np.isinf(rmse):
                    print(f"Epoch {epoch} 验证指标无效，跳过")
                    continue
                    
                # === 修改：在每个epoch后记录损失值 ===
                epoch_loss_record = {
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "split": split_idx+1
                }
                loss_history.append(epoch_loss_record)
                
                # === 修改：打印每个epoch的损失值 ===
                print(f"随机划分 {split_idx+1} Epoch {epoch+1}/{config.epochs} | "
                      f"训练损失: {train_loss:.6f} | "
                      f"验证损失: {val_loss:.6f} | "
                      f"RMSE: {rmse:.4f} | "
                      f"R²: {r2:.4f} | "
                      f"PCC: {pcc:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mse = mse
                    best_r2 = r2
                    best_pcc = pcc
                    
                    # 保存模型检查点
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_rmse': best_rmse,
                        'train_idx': train_idx,
                        'val_idx': val_idx
                    }
                    model_path = os.path.join(config.save_dir, f"random_split{split_idx+1}_best.pth")
                    torch.save(checkpoint, model_path)
                    print(f"已保存第{split_idx+1}次划分的最佳模型: {model_path}")
                    
                    # 记录最佳结果
                    best_results[f'split{split_idx+1}'] = {
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'pcc': pcc,
                        'samples': len(train_set),
                        'model_path': model_path
                    }
                    
                    # 更新日志文件
                    with open(log_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Split', 'MSE', 'R2', 'PCC', 'Num_Samples', 'Model_Path'])
                        
                        # 按划分顺序写入结果
                        for i in range(1, n_splits+1):
                            split_name = f'split{i}'
                            if split_name in best_results:
                                metrics = best_results[split_name]
                                writer.writerow([
                                    split_name,
                                    f"{metrics['mse']:.4f}",
                                    f"{metrics['r2']:.4f}",
                                    f"{metrics['pcc']:.4f}",
                                    metrics['samples'],
                                    metrics['model_path']
                                ])

                    patience = 30
                    print(f"随机划分 {split_idx+1} 已保存最佳模型 (epoch {epoch+1} | RMSE: {rmse:.4f} | R²: {r2:.4f})")
                else:
                    patience -= 1
                    if patience <= 0:
                        print(f"早停触发于epoch {epoch}")
                        break
                        
                if epoch % 10 == 0 and val_pred is not None:
                    plot_dir = os.path.join(config.save_dir, f"random_split_{split_idx+1}")
                    plot_results(val_true, val_pred, epoch, plot_dir)

                if config.wandb:
                    wandb.log({
                        f"split{split_idx+1}/train_loss": train_loss,
                        f"split{split_idx+1}/val_loss": val_loss,
                        f"split{split_idx+1}/val_rmse": rmse,
                        f"split{split_idx+1}/val_r2": r2,
                        "epoch": epoch
                    })
                    
            except Exception as e:
                print(f"训练异常: {str(e)}")
                continue

        if best_r2 is not None and best_pcc is not None:
            split_results.append({
                "split": split_idx+1,
                "best_rmse": best_rmse,
                "best_mse": best_mse,
                "r2": best_r2,
                "pcc": best_pcc,
                "num_samples": len(train_set)
            })
        else:
            print(f"随机划分 {split_idx+1} 无有效结果，跳过记录")
    
    # 计算并报告所有随机划分的平均结果
    if split_results:
        # 计算平均值
        best_rmses = [s['best_rmse'] for s in split_results]
        avg_rmse = np.mean(best_rmses)
        std_rmse = np.std(best_rmses)
        
        best_mses = [s['best_mse'] for s in split_results]
        avg_mse = np.mean(best_mses)
        std_mse = np.std(best_mses)
        
        r2s = [s['r2'] for s in split_results]
        avg_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)
        
        pccs = [s['pcc'] for s in split_results]
        avg_pcc = np.mean(pccs)
        std_pcc = np.std(pccs)
        
        # 将平均结果写入日志
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''])  # 空行分隔
            writer.writerow(['汇总统计'])
            writer.writerow(['RMSE (均值±标准差)', f"{avg_rmse:.4f} ± {std_rmse:.4f}"])
            writer.writerow(['MSE (均值±标准差)', f"{avg_mse:.4f} ± {std_mse:.4f}"])
            writer.writerow(['R² (均值±标准差)', f"{avg_r2:.4f} ± {std_r2:.4f}"])
            writer.writerow(['PCC (均值±标准差)', f"{avg_pcc:.4f} ± {std_pcc:.4f}"])
        
        print("\n" + "="*50)
        print(f"汇总统计: {n_splits}次随机划分平均结果")
        print(f"平均 MSE: {avg_mse:.4f} ± {std_mse:.4f}")
        print(f"平均 RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
        print(f"平均 R²: {avg_r2:.4f} ± {std_r2:.4f}")
        print(f"平均 PCC: {avg_pcc:.4f} ± {std_pcc:.4f}")
        print("="*50)
    else:
        print("所有随机划分均无有效结果！")
    
    # === 修改：在所有划分训练结束后生成损失历史CSV ===
    if loss_history:
        loss_csv_path = os.path.join(config.save_dir, f"loss_history_{timestamp}.csv")
        with open(loss_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['split', 'epoch', 'train_loss', 'val_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in loss_history:
                writer.writerow({
                    'split': record['split'],
                    'epoch': record['epoch'],
                    'train_loss': f"{record['train_loss']:.6f}",
                    'val_loss': f"{record['val_loss']:.6f}"
                })

        print(f"\n=== 训练损失历史已保存至: {loss_csv_path} ===")
    else:
        print("警告: 无损失历史记录")

def main():
    from config import config
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--train_csv", default=config.train_csv)
    parser.add_argument("--pdb_dir", default=config.pdb_dir)
    parser.add_argument("--prottrans_dir", default=config.prottrans_dir)
    parser.add_argument("--dssp_dir", default=config.dssp_dir)
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--contrast", action='store_true', default=config.contrast)
    parser.add_argument("--contrast_weight", type=float, default=config.contrast_weight)
    parser.add_argument("--temperature", type=float, default=config.temperature)
    parser.add_argument("--margin", type=float, default=config.margin)
    parser.add_argument("--drop_rate", type=float, default=config.drop_rate)
    parser.add_argument("--batch_size", type=int, default=1000)
    
    # 系统参数
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--num_workers", type=int, default=config.num_workers)
    parser.add_argument("--save_dir", default=config.save_dir)
    parser.add_argument("--fp16", action='store_true', default=config.fp16)
    parser.add_argument("--wandb", action='store_true', default=False)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_kfold(args)

if __name__ == "__main__":
    main()