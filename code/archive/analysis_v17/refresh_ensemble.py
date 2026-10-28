"""Rebuild same-seed equal-weight ensembles and data-dependent bilingual legends."""
from reconstruct import *
import subprocess,sys

def main():
 addon=OUT/'TIGER_ensemble_addendum';(addon/'figures').mkdir(exist_ok=True)
 for name in ['time_ensemble.py','validate_delivery.py']:shutil.copy2(Path(__file__).parent/'assets'/name,addon/'code'/name)
 subprocess.run([sys.executable,str(addon/'code/prepare_data.py')],cwd=OUT,check=True)
 t=read(addon/'tables/tiger_three_per_seed.csv');d=read(addon/'tables/rse_decomposition.csv');p=read(addon/'tables/paired_comparisons.csv')
 means=t.groupby('method')[METRICS].mean();dd=d.groupby('method').mean(numeric_only=True)
 delta=means.loc['TIGER-ensemble']-means.loc['TIGER-pair'];terms=dd.loc['TIGER-ensemble']-dd.loc['TIGER-pair']
 penalty=(means.loc['TIGER-single','RSE']-means.loc['TIGER-pair','RSE'])/2;diversity=penalty-delta.RSE
 assert abs(terms[['correlation_term','scale_term','bias_term']].sum()-delta.RSE)<1e-12
 stats_en='; '.join(f'{m} {delta[m]:+.6f}' for m in METRICS)
 counts=p[p.baseline=='TIGER-pair'].set_index('metric');ct='; '.join(f'{m}: {int(counts.loc[m,"seeds_improved"])}/10, Holm P={counts.loc[m,"p_holm_8"]:.6f}' for m in METRICS)
 discussion={
 'EN':f'The same-seed ensemble uses (calibrated P11 + calibrated S11)/2, with no fitted weights or further calibration. Ensemble-minus-P11 mean changes are {stats_en}. Improved seeds and eight-test Holm-adjusted P values are {ct}. Positive PCC/KCC differences and negative MAE/RSE differences indicate improvement. These paired-seed tests describe variability on this fixed external panel. For each family and seed, RSE = (1 − r²) + (c − r)² + b², where r is Pearson correlation, c = SD(prediction)/SD(observation), and b is the standardized mean bias. Averaging the decomposition over families and seeds gives changes {terms.correlation_term:+.6f}, {terms.scale_term:+.6f}, and {terms.bias_term:+.6f}, summing to {delta.RSE:+.6f}. The same equal-weight identity gives RSE(ensemble) = [RSE(P11)+RSE(S11)]/2 − sum[(P11−S11)²]/(4 SST). Relative to P11, the component-error penalty is {penalty:+.6f} and the diversity reduction is {diversity:.6f}. Complementary errors can improve ordering while scale compression or bias worsens amplitude agreement; improvement over the average component is not a guarantee of improvement over the better component [1]. Correlation is invariant to positive affine rescaling, whereas MAE and RSE depend on magnitude [2]. The decomposition supports a numerical explanation, not attribution to one training stage.',
 'ZH':f'逐种子集成为（已校准 P11 + 已校准 S11）/2，不拟合权重、不追加校准。集成减去 P11 的均值变化为 {stats_en}。改善种子数和八项统一 Holm 校正 P 值为 {ct}。PCC/KCC 正变化、MAE/RSE 负变化表示改善；检验只描述固定外测集的种子波动。对每个家族和种子，RSE = (1 − r²) + (c − r)² + b²，其中 r 为 Pearson 相关，c = SD(预测)/SD(实测)，b 为标准化均值偏差。先逐家族分解再平均，三项变化为 {terms.correlation_term:+.6f}、{terms.scale_term:+.6f}、{terms.bias_term:+.6f}，合计 {delta.RSE:+.6f}。等权平均还满足 RSE(集成) = [RSE(P11)+RSE(S11)]/2 − sum[(P11−S11)²]/(4 SST)。相对 P11，组分误差负担为 {penalty:+.6f}，多样性降低项为 {diversity:.6f}。互补误差可能改善排序，但尺度压缩或偏差可能恶化幅度一致性；优于组分平均误差并不保证优于较好组分 [1]。相关性对正仿射变换不变，而 MAE、RSE 依赖幅度 [2]。该分解提供数值解释，不能单独归因于某一训练阶段。'}
 intro={'EN':'V17 adopts the entire archived unsigned-quota-20,000 pair batch and reuses the V16 single head. Both original benchmark figures and the additional 24-predictor ensemble figures are supplied. No seed, weight or calibration is selected on the external panel.','ZH':'V17 整组采用既有 unsigned quota 20,000 配对模型，复用 V16 单肽模型；保留原 23 法基准，并额外提供含集成的 24 预测器图，不在外测集上选择种子、权重或校准。'}
 legends={}
 for m in METRICS:
  name='Fig4_benchmark_TIGER_ensemble'+('' if m=='PCC' else '_'+m)
  legends[name]={'EN':f'Twenty-four-predictor with-negative benchmark, {m}. Points show ten seeds; bars show two-sided 95% Student-t seed intervals. Families contribute equally. TIGER-ensemble is the within-seed average of the two calibrated components. Family heatmap cells are unscaled means; undefined correlations remain NA. '+discussion['EN'],
  'ZH':f'24 预测器含负样本基准，{m}。点为十种子，区间为双侧 95% Student-t 种子置信区间；家族等权。TIGER-ensemble 为同种子两个已校准组分的平均；家族热图为原始均值，未定义相关保留 NA。'+discussion['ZH']}
 for m in ['PCC','RSE']:
  name='Fig5_accuracy_cost_pareto_TIGER_ensemble'+('' if m=='PCC' else '_'+m)
  legends[name]={'EN':f'Accuracy–cost comparison including the fixed TIGER-ensemble, {m}. For each of the 23 backbones, both main endpoints use the training condition selected by the highest defined ten-seed mean PCC; the ensemble always uses P11/S11. This condition choice is descriptive external-test selection. Resident-model timing uses seed 123, one warm-up and 20 repetitions; P11 and the ensemble are remeasured in V17 and the other 22 methods retain their accepted timings. Structure generation is added only once. '+discussion['EN'],
  'ZH':f'加入固定 TIGER-ensemble 的准确性—成本图，{m}。23 骨干的两张正文图均采用最高已定义十种子 PCC 对应的训练条件，集成固定 P11/S11；该条件选择属于外测集描述性选择。驻留模型用种子 123、预热一次、重复 20 次；V17 重测 P11 和集成，其余 22 法沿用已验收计时，结构生成只计一次。'+discussion['ZH']}
 ref1='https://proceedings.neurips.cc/paper_files/paper/1994/file/b8c37e33defde51cf91e1e03e51657da-Paper.pdf';ref2='https://pubmed.ncbi.nlm.nih.gov/2868172/'
 dump(addon/'figure_legends.json',dict(version='V17',validation_only=FIXTURE,figures=legends,shared_mechanistic_legend=discussion,intro=intro,references=[ref1,ref2]))
 # The notebook imports this current data-only module; no obsolete V16 prose executes.
 (addon/'code/build_delivery.py').write_text('from pathlib import Path\nimport json\nDATA=json.loads((Path(__file__).resolve().parents[1]/"figure_legends.json").read_text())\nLEGENDS=DATA["figures"]\nDISCUSSION=DATA["shared_mechanistic_legend"]\nINTRO=DATA["intro"]\nREF1,REF2=DATA["references"]\n')
 (addon/'code/mechanistic_discussion.py').write_text('"""V17 discussion is generated from current results by refresh_ensemble.py."""\nfrom pathlib import Path\nimport json\ndef discussion(*args):\n return json.loads((Path(__file__).resolve().parents[1]/"figure_legends.json").read_text())["shared_mechanistic_legend"]\n')
 print('V17 ensemble data and conditional explanations prepared',flush=True)
if __name__=='__main__':main()
