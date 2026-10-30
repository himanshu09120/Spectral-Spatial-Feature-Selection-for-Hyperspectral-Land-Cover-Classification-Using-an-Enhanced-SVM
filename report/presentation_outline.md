# Eight-slide presentation outline

1. **Motivation and task:** agricultural mapping; supervised nine-class recognition; why reduce redundant spectral inputs.
2. **Dataset:** official LongKou source, inspected cube shape and measured class counts; distinguish background and labelled pixels. Add actual false-colour and ground-truth figures after generation.
3. **Challenges:** imbalance, spectral redundancy, small-object boundaries and random-pixel spatial dependence.
4. **Baselines:** E0 majority, E1 full-spectrum SVM, E2 PCA-SVM, E3 genuine noise-covariance MNF-SVM and E6 spatial Random Forest.
5. **FRSS-SVM:** Fisher score → correlation pruning → selected raw/mean/std features → training-only scaler → RBF SVM. Show 30 bands → 90 features as an example, not a measured selection.
6. **Experimental design:** shared stratified 70/15/15 split, 1,000/class cap, five seeds; validation macro-F1; held-out testing; optional separated block protocol.
7. **Measured results and ablation:** currently **pending; no training performed**. After evaluation, insert actual five-seed OA/F1, per-class confusion, E1/E4/E5 paired differences and runtime plots from saved artifacts. Do not insert illustrative scores.
8. **Conclusions and limitations:** implementation/data validation completed; performance hypothesis untested. After measurement, discuss observed gains/losses, spatial dependence, reproducibility, contributions and the public repository link.
