# Benchmark Methods

## Benchmark scope and method classification

We evaluated TIGER-pair and TIGER-single alongside 21 external predictors of antimicrobial potency, giving 23 model backbones. Each backbone was evaluated under with-negative and without-negative training protocols with seeds 120–129, yielding 460 method–condition–seed results. An additional TIGER-ensemble was constructed within matched seeds and was reported separately from the 23-backbone panel. The benchmark evaluates the implemented task predictors and disclosed regression adaptations, rather than the complete peptide-generation or screening systems of their source studies.

Method categories describe the information used by the evaluated implementations. EIPpred [1], AMPActiPred [2] and MBC-Attention [3] use engineered physicochemical representations; the first two are tree-based predictors. Das2021 [4], AMPlify [5], HydrAMP [6], EvoGradient [7], DeepAMP [8], APEX [11] and ANIA [20] form the sequence-model category without a pretrained protein language-model backbone. AMPLiT [10] combines sequence-derived and physicochemical features. AMP-Designer [9] uses a pretrained generative language model. LLAMP [12] and BERT-AmPEP60 [13] fine-tune language-model representations; HyperAMP [14], SenseXAMP-Ecoli [15], Deep-PIE [16], AMPGen [17], AMPpredictor [18] and tsAMP-CS [19] use frozen language-model features. GGN-ABPMIC [21] combines geometric graphs with language-model and secondary-structure features. TIGER combines global descriptors, sequence and structure-derived graphs. Categories are primary plotting labels, not mutually exclusive input inventories: SenseXAMP-Ecoli and AMPpredictor additionally use physicochemical information.

## Training corpora, labels and negative-exposure protocols

The external-predictor measured-MIC source comprised 5,737 records aggregated by arithmetic-mean MIC into 4,929 peptide sequences. The background-negative source contained 7,962 sequences; 13 exact conflicts with the measured pool were removed with measured labels taking precedence, leaving 7,949 sequences. Most regression adaptations assigned background negatives a pseudo-MIC of 10⁴ µg ml⁻¹. This pseudo-label represents the adopted training convention, not a measured MIC. Method-specific filters remained active, including the 40-residue limit for HyperAMP, removal of 39 negatives shorter than six residues for SenseXAMP-Ecoli, and structure availability for GGN-ABPMIC. EvoGradient used a distinct native µM target and an upper limit of 8,192 µM.

TIGER retained its sequence–inoculum training resource: 7,217 records representing 6,847 unique sequences, together with 7,860 background-negative sequences without exact overlap with the measured pool. Repeated sequences in different inoculum records were retained as separate observations and assigned to the same cross-validation fold. The sequence-minimum MIC distribution in Fig. 1b is a descriptive summary, not a replacement of all training rows by one minimum label. The external-predictor and TIGER corpora and label aggregation are therefore not identical.

For TIGER, the with-negative P11/S11 protocol includes both classification pretraining and negative exposure during regression. The without-negative P00/S00 protocol removes the complete classification-pretraining stage and regression negatives. The current TIGER-pair negative-exposure experiment compares only P00 with P11; intermediate single-stage-removal cells were not retrained for this reference. EvoGradient’s without-negative route also omits its classification stage. The condition comparisons consequently describe complete recorded protocols and do not isolate a universal single-factor effect of adding negative sequences.

## TIGER predictors and within-seed ensembles

TIGER-pair uses integer-encoded sequence inputs, ten global physicochemical descriptors, inoculum conditioning and a GATv2 structural graph. The reference activates 20 amino-acid-prior node channels and a 5 Å Cα contact adjacency; additional energy, coordinates and residue embeddings are separate interventions. Actual terminal-feature and external inoculum encodings were held fixed. Missing terminal annotations were not interpreted as experimental confirmation of free termini; the historical encoder does not recognize the N-terminal ACT alias as acetylation. All external peptides use the retained 10⁵–10⁶ CFU input code. Medium is retained as metadata, not supplied as an input feature.

Five folds were grouped by exact sequence, with the same saved fold indices across the ten training seeds. Measured–measured candidates were generated within the training rows using a ±5-residue window, at most 2,500 candidate visits per anchor and a directed NW score divided by anchor length of at least 0.7. Signed log₂-MIC differences were balanced using bins int(round(abs(Δ) × 100)), a quota of 20,000 per unsigned bin and a global cap of 250,000. The quota limits each bin, not the entire pair set. The reference retains neighbour-based validation with a 0.7 gate and top-50/fallback-20 construction; strict within-fold validation is a separate control. When enabled, classification pretraining lasts 15 epochs; regression runs for up to 70 epochs with patience 16. Five folds are trained sequentially within each seed, and saved fold-specific scaling and out-of-fold linear calibration are applied at inference.

TIGER-single shares the input backbone but regresses individual log₂ MIC values, which are subsequently differenced. Its single-peptide data path bypasses pair-bin balancing, so the complete existing ten-seed single-head batch was retained. Each seed-level TIGER predictor uses its five fold models. TIGER-ensemble is the arithmetic mean of the final calibrated pair and single predictions from the same seed; no ensemble weight or additional calibrator is fitted. This yields ten matched-seed ensembles, not ten replicates of a cross-seed average.

## Descriptor-based and non-language-model predictors

EIPpred uses a random-forest regressor on engineered peptide descriptors, whereas AMPActiPred uses a cascade-forest activity predictor. MBC-Attention applies a multi-branch convolutional network with attention to its physicochemical representation. The sequence-model group retains the registered Das2021 regression adaptation, attentive AMPlify regression head, DeepAMP and APEX task networks, and the ANIA inception–attention network. ANIA uses sequence-derived chaos-game and AAindex representations and was not classified as a pretrained language-model predictor. The main APEX and DeepAMP entries each use one retrained seed predictor; the original 40-checkpoint APEX model is not substituted for these repetitions.

HydrAMP is the registered continuous-MIC adaptation of its activity-discriminator backbone, rather than the full generative model. Sequences are padded to 50 residues and passed through a 128-dimensional embedding, a 64-filter convolution of width 16, max pooling of width 5, a 100-unit LSTM and a linear scalar output. The archived implementation pads the convolution by eight positions on the left and seven on the right. Adam (learning rate 0.001), MSE, batches of 64, a maximum of 100 epochs and validation-MSE patience 7 are used. The stopping-epoch weights, rather than a restored minimum-validation-loss checkpoint, are exported. Ten shuffled training folds are averaged within each seed.

EvoGradient combines CNN, Transformer, Attention and LSTM predictors with the released fixed weights 0.25000594, 0.25000460, 0.25000825 and 0.24998219, respectively. Native predictions are log₁₀ MIC in µM. Its with-negative route includes classification followed by regression; both stages use the registered validation-loss stopping and best-state restoration with patience 10. The formal archived Attention implementation supplies a residue-one/padding-zero Boolean mask directly as key_padding_mask. Under the called PyTorch API this masks residue keys rather than padding keys; this implementation limitation is disclosed rather than described as correctly masked attention. The subsequent corrected-mask and HydrAMP padding/checkpoint experiments are separate sensitivity experiments and do not replace the formal benchmark values reported here.

## Language-model, hybrid and structure-based predictors

LLAMP and BERT-AmPEP60 retain the language-model fine-tuning routes of their task drivers, whereas HyperAMP, SenseXAMP-Ecoli, Deep-PIE, AMPGen, AMPpredictor and tsAMP-CS use frozen language-model feature extractors with trained task heads. Frozen pretraining is not equivalent to random initialization: retraining here refers to the task-specific parameters and permitted fine-tuning. The registered feature routes include ESM, ankh, ProtBERT and ProtT5 representations. HyperAMP uses its 40-residue limit and attention-masked ankh features; GGN-ABPMIC uses attention-masked ProtT5 features, DSSP descriptors and geometric graphs from the project’s Rosetta-relaxed structure library. The corrected HyperAMP and GGN feature extractions and their retrained heads are the versions represented in the formal table.

SenseXAMP-Ecoli and AMPpredictor retain additional physicochemical inputs. SenseXAMP standardizes log-MIC labels using the complete training corpus before its seed-specific training/validation split; this includes the eventual validation subset but no external-test labels. tsAMP-CS uses the supplied E. coli ATCC 25922 strain representation. AMP-Designer retains its pretrained generative-language-model route and task-specific activity transformation, which is inverted before evaluation. Its main-table entry is a seed-specific task checkpoint, not a cross-seed ensemble.

AMPLiT uses the single registered regression implementation combining learned sequence representations and engineered descriptors. Word2Vec is fitted only on training-split sequences and exported as a fixed residue-vector table; external sequences are transformed with that table without refitting. Descriptor columns are aligned by saved feature names, short-sequence lag handling follows the saved schema, and the fitted target normalization is inverted at export. No alternate AMPLiT protocol is selected after inspecting external performance.

## Model selection and the statistical replicate

Optimization, splitting and checkpoint selection followed the actual method-specific drivers rather than a common early-stopping rule. For example, Das2021 restores the best validation-MSE state with patience 7. AMPlify uses validation-loss patience 50, minimum improvement 0.001 and best-weight restoration. MBC-Attention monitors training loss with patience 15 and retains the stopping weights. HyperAMP runs 20 epochs and restores its best validation checkpoint; SenseXAMP runs 80 epochs and exports final weights. AMPGen runs up to its configured 200-epoch limit and selects the best validation R² without an early-stopping break. Forest methods use their estimator-fitting procedures. All condition-specific checkpoints, configurations and inference transforms are retained for provenance.

One seed-level predictor is the unit of summary: ten fold models for HydrAMP, four weighted architecture components for EvoGradient, five fold models for TIGER, or the registered single task predictor for the other main entries. Matching integer seeds across methods or conditions does not guarantee identical validation splits or stochastic trajectories. The TIGER-pair reference was selected from an earlier ablation batch after examination of this same external panel. Subsequent ablations and condition-selected comparisons are therefore exploratory rather than independent prospective validation.

## External endpoint and concentration harmonization

All predictors were evaluated on the same 208 peptides from 11 families and the same 2,211 within-family pairs. Each unordered pair retained one recorded anchor–query orientation. Pair labels were defined as Δ = log₂ MIC(anchor) − log₂ MIC(query), so positive values indicate a more potent query. External pairs were retained at directed NW score divided by anchor length ≥0.3. The maximum training-proximity scores shown in Fig. 2 instead divide by the longer sequence length; NW scores are not percentage identities. There was no exact test–training sequence overlap. Within-family experimental conditions were confirmed consistent; apparent metadata discrepancies did not trigger additional edge deletion.

Absolute-potency predictions were expressed in log₁₀ MIC in µg ml⁻¹ before subtraction; their pair differences were divided by log₁₀(2) to obtain log₂ units. EvoGradient’s native log₁₀ MIC in µM was converted once per peptide as log₁₀ MIC(µg ml⁻¹) = log₁₀ MIC(µM) + log₁₀[MW(Da)/1,000]. Molecular mass was computed with the driver’s residue-mass table plus 18.0152 Da for water, without inferring terminal modifications. Because this correction is peptide-specific, it can change correlations as well as errors. TIGER-pair’s calibrated log₂ differences were evaluated directly. Every prediction vector was aligned by family, anchor sequence and query sequence and checked for complete, finite coverage of the canonical endpoint.

## Metrics, uncertainty and paired comparisons

Within each family, we calculated Pearson correlation (PCC), tie-corrected Kendall τb (KCC), mean absolute error of log₁₀-MIC differences (log10MAE) and relative squared error (RSE). For log₂ pair labels y and predictions ŷ, log10MAE = mean(|ŷ − y|) × log₁₀(2), and RSE = Σ(ŷ − y)² / Σ(y − mean(y))². RSE = 1 corresponds to predicting the family mean. Family metrics were averaged with equal weights across all 11 families before averaging the ten seeds. This avoids weighting the result by unequal family pair counts.

A family correlation is undefined for constant predictions. Undefined values were propagated to the seed macro and ten-seed correlation mean, without deleting the family, dropping a seed or assigning zero. In the formal AMPLiT with-negative predictions, seed 122 has constant output within Hydrodamin-3; its PCC and KCC summaries therefore remain undefined, whereas error metrics remain defined. A separately labelled zero-contribution sensitivity is not used for primary ranking or Pareto selection.

Across-seed dispersion is reported as sample SD. Confidence intervals are two-sided Student t intervals with nine degrees of freedom. Paired effects use differences matched by seed, with unadjusted t intervals and two-sided Wilcoxon signed-rank tests. The 41 ablation controls across four metrics form one family of 164 Holm-adjusted tests. Benchmark with-minus-without comparisons use 23 tests within each metric; the eight ensemble–component tests across four metrics are adjusted jointly. All-zero contrasts receive P = 1, and undefined tests retain their multiplicity slot. The same P00–P11 contrast therefore has different adjusted P values in the ablation and benchmark families; these contexts are reported explicitly. Shared peptide endpoints are not treated as independent biological replicates.

## Inference timing and Pareto construction

Inference cost is the accepted median of 20 complete-workload repetitions after one warm-up, using the actual with-negative seed-123 predictor. Models are loaded before timing and remain resident. As applicable, the timed region includes query encoding, live language-model or descriptor extraction, graph construction, fold-specific standardization, model prediction, inverse target transforms, calibration and construction of all 2,211 pair differences. Identity, finite-value, independent pair-algebra and archived-prediction checks are performed outside the clock. EIPpred and AMPActiPred use CPU; the other recorded predictors use an RTX 4090 D. AMPlify uses its validated TensorFlow eager-GPU route and MBC-Attention its graph-GPU route. The comparison therefore retains recorded device differences.

The current unsigned-quota-20,000 TIGER-pair and its ensemble were timed separately; the other 22 backbones retain accepted measurements from the same defined workload. TIGER-pair takes 1.958970 s and the ensemble 3.968514 s with structures ready; the retained TIGER-single time is 1.965852 s. Structure generation is represented separately by a serial estimate: the median paired-stage generation time of 1.5775 s measured for 30 peptides multiplied by 208, giving 328.12 s. This increment is added once to each structural pipeline, including an ensemble that shares the structure library; it is not a direct end-to-end measurement for all 208 peptides.

Within a backbone, the two negative-exposure conditions share the with-negative time coordinate because the architecture and parameter count agree; numerical weights may differ, and this plotting convention does not establish identical independently measured latency. In the main PCC and RSE views, one complete condition per backbone is chosen by the highest defined ten-seed PCC, and both metrics use that same condition. AMPLiT contributes the only condition with a defined PCC. This is descriptive test-set selection. Supplementary plots show both conditions, connect their means and omit structures-ready points; their frontiers use all displayed conditions. A point is nondominated if no displayed alternative is both no slower and at least as accurate, with a strict advantage in one dimension. PCC is maximized and RSE minimized. Connecting segments are visual guides rather than evaluated intermediate models.

## References

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
