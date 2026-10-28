# 基准测试方法

## 基准范围与方法分类

我们将TIGER-pair和TIGER-single与21种外部抗菌活性预测器比较，共23个模型骨干。每个骨干均评估含负样本和不含负样本两种训练方案，使用种子120–129，共460个方法—条件—种子结果。另按匹配种子构建TIGER-ensemble，并在23骨干主基准之外单独展示。基准评价实际任务预测器及明确披露的回归适配，不评价来源研究完整的多肽生成或筛选系统。

方法类别依据实际实现所使用的信息。EIPpred [1]、AMPActiPred [2]和MBC-Attention [3]采用工程化理化表征，前两者为树模型。Das2021 [4]、AMPlify [5]、HydrAMP [6]、EvoGradient [7]、DeepAMP [8]、APEX [11]和ANIA [20]归入不使用预训练蛋白语言模型骨干的序列模型。AMPLiT [10]结合序列衍生与理化特征；AMP-Designer [9]采用预训练生成式语言模型；LLAMP [12]和BERT-AmPEP60 [13]微调语言模型；HyperAMP [14]、SenseXAMP-Ecoli [15]、Deep-PIE [16]、AMPGen [17]、AMPpredictor [18]和tsAMP-CS [19]使用冻结语言模型特征。GGN-ABPMIC [21]结合几何图、语言模型及二级结构特征；TIGER结合全局描述子、序列和结构衍生图。分类是绘图的主要标签，不是互斥的输入清单：SenseXAMP-Ecoli和AMPpredictor还使用理化信息。

## 训练语料、标签与负样本暴露方案

外部预测器的实测MIC来源包含5,737条记录，按序列取MIC算术均值后形成4,929条序列。背景负样本源含7,962条序列；与实测池完全相同的13条冲突序列按实测标签优先原则移除，剩余7,949条。多数回归适配为背景负样本赋予10⁴ µg ml⁻¹的伪MIC；该值是训练约定，而非实测MIC。各方法保留自身过滤规则，包括HyperAMP的40残基限制、SenseXAMP-Ecoli排除39条短于六残基的负序列，以及GGN-ABPMIC的结构可用性筛选。EvoGradient采用独立的原生µM目标，并设8,192 µM上限。

TIGER保留序列—接种量训练资源：7,217条记录、6,847条唯一序列，以及与实测池无完全相同序列的7,860条背景负样本。不同接种量记录中的重复序列保留为独立观察，并被分入同一交叉验证折。图1b按序列取最小MIC的分布仅为描述性汇总，不表示训练中将全部行替换为每序列一个最小标签。因此，外部预测器与TIGER的语料及标签聚合并不相同。

对TIGER，含负样本P11/S11方案同时包含分类预训练和回归负样本暴露；无负样本P00/S00方案移除整个分类预训练阶段及回归负样本。当前TIGER-pair负样本实验仅比较P00与P11，未在新参考下重新训练仅移除一个阶段的中间组。EvoGradient无负样本路线同样省略分类阶段。因此，条件对照描述完整的已记录训练方案，不能普遍隔离仅添加负序列这一单因素效应。

## TIGER预测器与种子内集成

TIGER-pair采用整数编码序列、十维全局理化描述子、接种量条件及GATv2结构图。参考启用20维氨基酸先验节点通道和5 Å Cα接触邻接；能量、坐标和残基嵌入添加属于独立干预。实际末端特征及外测接种量编码保持固定。末端注释缺失不被解释为已实验确认的自由端；历史编码器不将N端ACT别名识别为乙酰化。全部外测肽保留10⁵–10⁶ CFU输入编码。培养基仅保留为元数据，不作为模型输入。

五折按精确序列分组，十个训练种子使用相同的已保存折索引。实测—实测候选仅在训练行中构建，采用±5残基窗口、每锚肽最多访问2,500个候选，并要求NW分数除以锚肽长度至少为0.7。有符号log₂ MIC差值按int(round(abs(Δ) × 100))分箱，以每个无符号箱20,000的配额和250,000全局上限平衡。配额限制每个分箱，而非整个配对集。主参考保留门限0.7及top-50/fallback-20的邻居验证构造；严格折内验证为独立对照。启用时分类预训练为15轮，回归最多70轮、patience为16。每种子内五折顺序训练，推断应用保存的折特异标准化及折外线性校准。

TIGER-single共享输入骨干，但回归单肽log₂ MIC后再相减。其单肽数据路径不经过配对分箱平衡，因此保留已有完整十种子单头模型。每个种子级TIGER预测器采用对应五折模型。TIGER-ensemble为同一种子最终已校准配对与单肽预测的算术平均，不拟合集成权重或额外校准器。由此得到十个同种子集成，而非把跨种子平均重复十次。

## 描述子模型与非语言模型预测器

EIPpred采用基于工程化肽描述子的随机森林回归器，AMPActiPred采用级联森林活性预测器。MBC-Attention对理化表征应用多分支卷积与注意力网络。序列模型组保留已登记的Das2021回归适配、AMPlify注意力回归头、DeepAMP和APEX任务网络，以及ANIA的inception—attention网络。ANIA采用序列衍生的混沌博弈及AAindex表征，不归类为预训练语言模型预测器。主表中的APEX和DeepAMP各使用一个重训种子预测器，不以原始40检查点APEX替换这些重复。

HydrAMP为其活性判别器骨干的连续MIC回归适配，而非完整生成模型。序列填充至50残基，依次经过128维嵌入、64通道且宽度16的卷积、宽度5的最大池化、100单元LSTM及线性标量输出。归档实现的卷积填充为左8、右7。采用Adam（学习率0.001）、MSE、批量64、最多100轮及验证MSE patience 7；导出停止轮权重，而非恢复最低验证损失检查点。每种子内平均十个打乱划分的折模型。

EvoGradient以发布的固定权重0.25000594、0.25000460、0.25000825和0.24998219组合CNN、Transformer、Attention和LSTM预测器，原生输出为µM单位的log₁₀ MIC。含负样本路线先分类再回归，两阶段采用登记的验证损失早停和最佳状态恢复，patience为10。正式归档Attention实现将“残基1、填充0”的布尔掩码直接传入key_padding_mask；在其调用的PyTorch接口中，这会屏蔽残基键而非填充键。此实现局限予以明确披露，不描述为正确的填充掩码。随后完成的掩码修复及HydrAMP填充／检查点实验属于独立敏感性实验，未替换此处正式基准数值。

## 语言模型、混合及结构预测器

LLAMP和BERT-AmPEP60保留任务驱动中的语言模型微调路线；HyperAMP、SenseXAMP-Ecoli、Deep-PIE、AMPGen、AMPpredictor和tsAMP-CS采用冻结语言模型特征提取器及训练后的任务头。冻结预训练并不等同于随机初始化；此处“重训”指任务参数和允许的微调。登记的特征路线包括ESM、ankh、ProtBERT及ProtT5表征。HyperAMP保留40残基限制并使用注意力掩码后的ankh特征；GGN-ABPMIC结合掩码后的ProtT5特征、DSSP描述子及项目Rosetta松弛结构库构建的几何图。正式表采用修正后的HyperAMP、GGN特征提取及相应重训任务头。

SenseXAMP-Ecoli与AMPpredictor保留额外理化输入。SenseXAMP在种子特异的训练／验证划分之前，用完整训练语料标准化log-MIC标签，因此包含最终验证子集，但不使用外测标签。tsAMP-CS采用提供的E. coli ATCC 25922菌株表征。AMP-Designer保留预训练生成式语言模型路线及任务特异活性变换，并在评估前逆变换；主表为种子特异任务检查点，而非跨种子集成。

AMPLiT采用唯一登记的回归实现，结合学习的序列表征和工程化描述子。Word2Vec只在训练子集序列上拟合并导出固定残基向量表；外测序列通过该表变换，不重新拟合。描述子按保存的特征名对齐，短序列滞后特征沿用保存的处理规则，导出时逆转已拟合的目标标准化。不根据外测表现再选择其他AMPLiT方案。

## 模型选择与统计重复单位

优化、划分及检查点选择遵循实际方法驱动，不采用统一早停规则。例如，Das2021按验证MSE、patience 7恢复最佳状态；AMPlify采用验证损失patience 50、最小改善0.001及最佳权重恢复；MBC-Attention监控训练损失、patience 15并保留停止权重；HyperAMP运行20轮后恢复最佳验证检查点；SenseXAMP运行80轮并导出末轮权重；AMPGen最多运行配置的200轮，按最佳验证R²选取状态而无早停退出；森林模型采用各自估计器拟合流程。所有条件特异的检查点、配置及推断变换均保留用于溯源。

每个种子级预测器为汇总单位：HydrAMP含十折模型，EvoGradient含四个加权架构组分，TIGER含五折模型，其他主表条目采用各自登记的单个任务预测器。不同方法或条件使用相同整数种子，不保证验证划分或随机轨迹完全一致。TIGER-pair参考是在查看同一外测集后从此前消融批次中选定，因此后续消融和按条件择优比较属于探索性分析，而非独立前瞻验证。

## 外测终点与浓度单位统一

全部预测器在相同的11家族、208肽及2,211个家族内配对上评估。每个无序对保留一个记录的锚肽—查询肽方向；标签定义为Δ = log₂ MIC（锚肽）− log₂ MIC（查询肽），正值表示查询肽活性更强。外测配对要求NW分数除以锚肽长度≥0.3。图2的最大训练邻近分数则除以两序列中的较长长度；NW分数不是序列一致性百分比。外测与训练序列无完全相同序列。家族内实验条件已确认一致，表观元数据差异未导致额外删边。

绝对活性预测在相减前统一为µg ml⁻¹单位的log₁₀ MIC，再将配对差值除以log₁₀(2)得到log₂单位。EvoGradient原生µM输出逐肽仅换算一次：log₁₀ MIC(µg ml⁻¹) = log₁₀ MIC(µM) + log₁₀[MW(Da)/1,000]。分子量采用驱动的残基质量表并加18.0152 Da水，不推断末端修饰。该修正具有肽特异性，因此既可改变误差，也可改变相关。TIGER-pair的已校准log₂差值直接评估。所有预测按家族、锚肽序列和查询肽序列对齐，并核验规范终点的完整覆盖与有限值。

## 指标、不确定性与配对比较

在每个家族内计算Pearson相关（PCC）、含并列值校正的Kendall τb（KCC）、log₁₀ MIC差值的平均绝对误差（log10MAE）及相对平方误差（RSE）。对log₂配对标签y和预测ŷ，log10MAE = mean(|ŷ − y|) × log₁₀(2)，RSE = Σ(ŷ − y)² / Σ(y − mean(y))²；RSE=1对应预测该家族均值。先对11个家族等权平均，再汇总十个种子，避免按不同家族的配对数量加权结果。

常数预测的家族相关未定义。未定义值传播到种子宏平均及十种子相关均值，不删除家族、不丢弃种子，也不置零。正式AMPLiT含负样本预测中，种子122在Hydrodamin-3内输出常数，因此PCC和KCC汇总保持未定义，误差指标仍可计算。另行标记的零贡献敏感性结果不用于主要排名或Pareto选型。

种子间离散度报告样本标准差，置信区间采用自由度9的双侧Student t区间。配对效应按相同种子作差，报告未校正t区间并采用双侧Wilcoxon符号秩检验。41个消融对照与四指标统一构成164项Holm校正检验族；基准含负减无负比较在每指标内校正23项检验；四指标的八项集成—组分检验联合校正。全零差异赋P=1，未定义检验保留其多重比较槽位。因此同一P00–P11对照在消融族和基准族中具有不同校正P值，报告中明确区分范围。共享肽端点不作为独立生物学重复。

## 推断计时与Pareto构建

推断成本为实际含负样本种子123预测器预热一次后，20次完整工作负荷重复的已验收中位数。模型在计时前加载并保持常驻。按方法适用情况，计时区包含查询编码、实时语言模型或描述子提取、构图、折特异标准化、预测、目标逆变换、校准及全部2,211个配对差值构造；身份、有限值、独立配对代数及归档预测对账在计时区外完成。EIPpred和AMPActiPred使用CPU，其他记录预测器使用RTX 4090 D；AMPlify采用已验证的TensorFlow eager GPU路线，MBC-Attention采用graph GPU路线。比较因此保留实际设备差异。

当前每箱20,000的无符号配额TIGER-pair及其集成分别重测，其余22骨干保留同一已定义工作负荷的已验收测量。结构已备时，TIGER-pair耗时1.958970秒，集成为3.968514秒，保留的TIGER-single耗时为1.965852秒。结构生成单独采用串行估计：30条实测肽两阶段生成时间之和的中位数1.5775秒乘以208，得到328.12秒。每个结构流程只增加一次该增量，包括共享结构库的集成；它不是对全部208肽的直接端到端测量。

同一骨干两种负样本条件共享含负样本计时横坐标，依据是架构和参数量相同；数值权重可不同，这一绘图约定不证明独立实测延迟完全一致。正文PCC和RSE图均按最高可定义十种子PCC为每骨干选择一个完整条件，两指标展示同一条件；AMPLiT采用唯一可定义PCC条件。这属于外测集上的描述性选择。附图显示两条件并连线，取消结构已备点，其前沿使用全部显示条件。若不存在同时不更慢、性能不更差且至少一维严格更优的显示方案，则该点为非支配点；PCC最大化，RSE最小化。连接线仅作视觉引导，不代表已经评估的中间模型。

## 参考文献

1. Bajiya, N. et al. Prediction of inhibitory peptides against E. coli with desired MIC value. Sci. Rep. 15, 4672 (2025). https://doi.org/10.1038/s41598-025-86638-z

2. Yao, L. et al. AMPActiPred: a three-stage framework for predicting antibacterial peptides and activity levels. Protein Sci. 33, e5006 (2024). https://doi.org/10.1002/pro.5006

3. Yan, J. et al. A deep learning method for predicting the minimum inhibitory concentration of antimicrobial peptides against E. coli using Multi-Branch-CNN and Attention. mSystems 8, e00345-23 (2023). https://doi.org/10.1128/msystems.00345-23

4. Das, P. et al. Accelerated antimicrobial discovery via deep generative models and molecular dynamics simulations. Nat. Biomed. Eng. 5, 613–623 (2021). https://doi.org/10.1038/s41551-021-00689-x

5. Li, C. et al. AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens. BMC Genomics 23, 77 (2022). https://doi.org/10.1186/s12864-022-08310-4

6. Szymczak, P. et al. Discovering highly potent antimicrobial peptides with deep generative model HydrAMP. Nat. Commun. 14, 1453 (2023). https://doi.org/10.1038/s41467-023-36994-z

7. Wang, B. et al. Explainable deep learning and virtual evolution identifies antimicrobial peptides with activity against multidrug-resistant human pathogens. Nat. Microbiol. 10, 332–347 (2025). https://doi.org/10.1038/s41564-024-01907-3

8. Pandi, A. et al. Cell-free biosynthesis combined with deep learning accelerates de novo development of antimicrobial peptides. Nat. Commun. 14, 7197 (2023). https://doi.org/10.1038/s41467-023-42434-9

9. Wang, J. et al. Discovery of antimicrobial peptides with notable antibacterial potency by an LLM-based foundation model. Sci. Adv. 11, eads8932 (2025). https://doi.org/10.1126/sciadv.ads8932

10. Chen, S. et al. Identification of antimicrobial peptides from ancient gut microbiomes. Nat. Commun. 17, 1788 (2026). https://doi.org/10.1038/s41467-026-68495-0

11. Wan, F. et al. Deep-learning-enabled antibiotic discovery through molecular de-extinction. Nat. Biomed. Eng. 8, 854–871 (2024). https://doi.org/10.1038/s41551-024-01201-x

12. Bae, D. et al. AI-guided discovery and optimization of antimicrobial peptides through species-aware language model. Brief. Bioinform. 26, bbaf343 (2025). https://doi.org/10.1093/bib/bbaf343

13. Cai, J. et al. BERT-AmPEP60: a BERT-based transfer learning approach to predict the minimum inhibitory concentrations of antimicrobial peptides for E. coli and S. aureus. J. Chem. Inf. Model. 65, 3186–3202 (2025). https://doi.org/10.1021/acs.jcim.4c01749

14. Dong, R. et al. Painting peptides with antimicrobial potency through deep reinforcement learning. Adv. Sci. 12, e06332 (2025). https://doi.org/10.1002/advs.202506332

15. Zhang, W. et al. Fuse feeds as one: cross-modal framework for general identification of AMPs. Brief. Bioinform. 24, bbad336 (2023). https://doi.org/10.1093/bib/bbad336

16. Abbasi, M. et al. A novel deep learning framework for predicting antimicrobial peptide activity using ProtBert and neural networks. In Lecture Notes in Computer Science (IbPRIA 2025) 71–82 (2025). https://doi.org/10.1007/978-3-031-99568-2_6

17. Jin, S. et al. AMPGen: an evolutionary information-reserved and diffusion-driven generative model for de novo design of antimicrobial peptides. Commun. Biol. 8, 839 (2025). https://doi.org/10.1038/s42003-025-08282-7

18. Dong, R. et al. Exploring the repository of de novo-designed bifunctional antimicrobial peptides through deep learning. eLife 13, RP97330 (2025). https://doi.org/10.7554/eLife.97330.3

19. Li, H. et al. tsAMP: a strain-level antimicrobial peptide identification framework based on large language models and pathogen genomes. Front. Microbiol. 17, 1842380 (2026). https://doi.org/10.3389/fmicb.2026.1842380

20. Chiu, Y.-P. et al. ANIA: an inception-attention network for predicting minimum inhibitory concentration of antimicrobial peptides. Brief. Bioinform. 27, bbag023 (2026). https://doi.org/10.1093/bib/bbag023

21. Bin, Y. et al. Prediction of minimum inhibitory concentration of antibacterial peptides using geometric graph networks and dynamic modeling. Brief. Bioinform. 27, bbag170 (2026). https://doi.org/10.1093/bib/bbag170
