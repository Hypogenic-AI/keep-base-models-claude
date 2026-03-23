# Downloaded Papers

## Core KL/f-Divergence in Alignment

1. **Direct Preference Optimization** (rafailov2023_dpo.pdf)
   - Authors: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn
   - Year: 2023 | arXiv: 2305.18290
   - Why relevant: Foundation method with implicit KL regularization

2. **RL with KL penalties is better viewed as Bayesian inference** (korbak2022_kl_bayesian_inference.pdf)
   - Authors: Korbak, Perez, Buckley
   - Year: 2022 | arXiv: 2205.11275
   - Why relevant: Theoretical foundation for KL-constrained alignment as variational inference

3. **Beyond Reverse KL: f-DPO** (wang2023_f_dpo.pdf)
   - Authors: Wang, Jiang, Yang, Liu, Chen
   - Year: 2023 | arXiv: 2309.16240
   - Why relevant: Generalizes DPO to arbitrary f-divergences

4. **Aligning LMs with Preferences through f-divergence Minimization** (go2023_f_divergence_alignment.pdf)
   - Authors: Go, Korbak, Kruszewski, Rozen, Ryu, Dymetman
   - Year: 2023 | arXiv: 2302.08215
   - Why relevant: f-DPG framework, divergence-diversity tradeoffs

5. **Token-level Direct Preference Optimization** (zeng2024_token_dpo.pdf)
   - Authors: Zeng, Liu, Ma, Yang, Zhang, Wang
   - Year: 2024 | arXiv: 2404.11999
   - Why relevant: Token-level forward KL constraints improve diversity

6. **Correcting the Mythos of KL-Regularization** (huang2024_chi_squared.pdf)
   - Authors: Huang, Zhan, Xie, Lee, Sun, Krishnamurthy, Foster
   - Year: 2024 | arXiv: 2407.13399
   - Why relevant: Chi-squared divergence as alternative to KL

## Diversity & Alignment Tax

7. **Understanding the Effects of RLHF on LLM Generalisation and Diversity** (kirk2023_rlhf_diversity.pdf)
   - Authors: Kirk, Mediratta, Nalmpantis, Luketina, Hambro, Grefenstette, Raileanu
   - Year: 2023 | arXiv: 2310.06452
   - Why relevant: First rigorous empirical study of RLHF mode collapse

8. **Mitigating the Alignment Tax of RLHF** (lin2023_alignment_tax.pdf)
   - Authors: Lin, Lin, Xiong, Diao, et al.
   - Year: 2023 | arXiv: 2309.06256
   - Why relevant: Model averaging to reduce alignment tax

9. **Creativity Has Left the Chat** (mohammadi2024_creativity_debiasing.pdf)
   - Authors: Mohammadi
   - Year: 2024 | arXiv: 2406.05587
   - Why relevant: Documents creativity loss from alignment/debiasing

10. **One fish, two fish, but not the whole sea** (murthy2024_conceptual_diversity.pdf)
    - Authors: Murthy, Ullman, Hu
    - Year: 2024 | arXiv: 2411.04427
    - Why relevant: Measures conceptual diversity reduction from alignment

## Base Model Distribution Preservation & Elasticity

11. **Language Models Resist Alignment** (ji2024_resist_alignment.pdf)
    - Authors: Ji, Wang, Qiu, Chen, Zhou, Li, Lou, Dai, Liu, Yang
    - Year: 2024 | arXiv: 2406.06144
    - Why relevant: Demonstrates "elasticity" — base distributions persist through alignment

12. **The Unlocking Spell on Base LLMs** (lin2023_unlocking_base.pdf)
    - Authors: Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula, Choi
    - Year: 2023 | arXiv: 2312.01552
    - Why relevant: Argues alignment is superficial; base capabilities accessible via ICL

13. **From Distributional to Overton Pluralism** (lake2024_distributional_pluralism.pdf)
    - Authors: Lake, Choi, Durrett
    - Year: 2024 | arXiv: 2406.17692
    - Why relevant: Alignment doesn't suppress useful information, just aggregates it

14. **Self-Distillation Bridges Distribution Gap** (yang2024_self_distillation.pdf)
    - Authors: Yang, Liu, Pang, Wang, Feng, Zhu, Chen
    - Year: 2024 | arXiv: 2402.13669
    - Why relevant: Bridges distribution gap to preserve capabilities during fine-tuning

## Model Interpolation & Merging

15. **Online Merging Optimizers** (lu2024_online_merging.pdf)
    - Authors: Lu, Yu, Huang, Fan, Lin, Zhou
    - Year: 2024 | arXiv: 2405.17931
    - Why relevant: Online weight merging during RLHF preserves SFT capabilities

16. **Adding Alignment Control to Language Models** (zhu2025_alignment_control.pdf)
    - Authors: Zhu, Zhang, Wang
    - Year: 2025 | arXiv: 2503.04346
    - Why relevant: Interpolation coefficient for continuous alignment control

17. **Emulated Disalignment** (zhou2024_emulated_disalignment.pdf)
    - Authors: Zhou, Liu, Dong, Liu, Yang, Ouyang, Qiao
    - Year: 2024 | arXiv: 2402.12343
    - Why relevant: Distribution arithmetic proves base distributions survive alignment

## Inference-Time Alignment

18. **BoNBoN Alignment** (gui2024_bonbon.pdf)
    - Authors: Gui, Garbacea, Veitch
    - Year: 2024 | arXiv: 2406.00832
    - Why relevant: Best-of-N as KL-efficient alignment preserving base distribution

19. **Asymptotics of Language Model Alignment** (yang2024_asymptotics_alignment.pdf)
    - Authors: Yang, Salamatian, Sun, Suresh, Beirami
    - Year: 2024 | arXiv: 2404.01730
    - Why relevant: Closed-form analysis of KL-constrained alignment solutions

## Distribution Learning & Diverse Optimization

20. **Alignment as Distribution Learning** (yun2025_alignment_distribution.pdf)
    - Authors: Yun, Kim, Park, Kim, Ryu, Cho, Jun
    - Year: 2025 | arXiv: 2506.01523
    - Why relevant: Distribution learning framework avoiding degeneracy

21. **Diverse Preference Optimization** (lanchantin2025_diverse_pref_opt.pdf)
    - Authors: Lanchantin, Chen, Dhuliawala, Yu, Weston, Sukhbaatar, Kulikov
    - Year: 2025 | arXiv: 2501.18101
    - Why relevant: Preference selection for diversity while maintaining quality

22. **Creative Preference Optimization** (ismayilzada2025_creative_pref_opt.pdf)
    - Authors: Ismayilzada, Laverghetta, Luchini, Patel, Bosselut, van der Plas, Beaty
    - Year: 2025 | arXiv: 2505.14442
    - Why relevant: Optimization for creative output preservation

## RLHF Foundations

23. **Training language models to follow instructions with human feedback** (ouyang2022_instructgpt.pdf)
    - Authors: Ouyang, Wu, Jiang, Almeida, et al.
    - Year: 2022 | arXiv: 2203.02155
    - Why relevant: Original InstructGPT RLHF approach

24. **Training a Helpful and Harmless Assistant with RLHF** (bai2022_helpful_harmless.pdf)
    - Authors: Bai, Jones, Ndousse, Askell, et al.
    - Year: 2022 | arXiv: 2204.05862
    - Why relevant: Anthropic's Constitutional AI with KL-reward analysis
