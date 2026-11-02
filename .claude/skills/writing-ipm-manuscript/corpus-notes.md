# IP&M corpus notes

Read 2026-11-02 from arXiv accepted versions that carry an IP&M journal reference. Text extracted with pdftotext.

| arXiv | Paper | IP&M ref | Layout | Abstract words |
|---|---|---|---|---|
| 2603.09152 | DataFactory: multi-agent framework for TableQA | 63(6):104723, 2026 | Elsevier `cas-sc`, Highlights page, ARTICLE INFO / ABSTRACT block, 33 pp | about 240 |
| 2605.17630 | SegRAG: retrieval-augmented spatial prompting | 64(2), 105190, 2026 | Elsevier single column, 37 pp | 281 |
| 2409.07507 | Traceable LLM-based validation of KG statements | 62(4):104128, 2025 | Author-formatted accepted version, numbered sections | 189 |
| 2411.01952 | Estimating quality of medical research with ChatGPT | 2025, to appear | Plain, unnumbered Introduction / Methods / Results / Discussion / Conclusion | 264 |
| 2506.00467 | SST: self-training with self-adaptive thresholding | 62(5):104158, 2025 | Elsevier `cas-sc`, ARTICLE INFO block | about 320 |
| 2508.16013 | Political ideology shifts in LLMs | 2026, 105124 | Elsevier, 87 pp with appendices | not counted (abstract word counts above are approximate, from pdftotext) |

## Structure seen in DataFactory (2603.09152)

1 Introduction; 2 Related Work (2.1 LLM-based methods, 2.2 agent-based methods); 3 Research Objectives (five numbered objectives); 4 Methodology; 5 Evaluation (5.1 setup: benchmarks, LLMs, baselines; 5.2-5.6 one subsection per RQ1-RQ5, each ending "RQn Summary Answer" in a box); 6 Theoretical and Practical Implications; 7 Conclusion and Future Work; Appendix A; Acknowledgments; CRediT; References.

## Structure seen in SegRAG (2605.17630)

1 Introduction; 2 Research Objectives (RO1-RO4, then a paragraph stating the main contribution and three differences from prior work); 3 Related Work and Research Gap; 4 Methodology; 5 Datasets and Experimental Protocol; 6 Results; 7 Ablation Analysis; 8 Discussion and Implications (8.3 Information-management and practical implications, 8.4 Scope, robustness, and external validity); 9 Conclusion; Acknowledgements; Author Contributions; Declaration of Competing Interest; Data Availability (datasets, split manifests, code URL).

## Structure seen in Adam & Kliegr (2409.07507)

Numbered sections; 7.5 Limitations and future work sits at the end of the discussion; CRediT, Data availability, Acknowledgements follow.

## Register samples (paraphrased patterns, not text to copy)

- Abstract opening names the limitation of current approaches, then "This paper introduces X, a ...".
- Abstract closes with a practical-use sentence ("offers design guidelines ...", "results suggest ... could be used for ...").
- Introduction ends the framework description with "In summary, our contributions are: (1) ...; (2) ...".
- Scope sentence in Discussion: "The experiments establish X for ...; separate evaluation remains necessary for ...".
- Cost trade-offs stated openly ("Despite higher token costs ...", "incurs offline construction and storage costs").

## Not verified

The Guide for Authors was blocked (HTTP 403 and an error page on ScienceDirect, 403 on Elsevier and SciSpace mirrors). Unverified: Highlights character limit, abstract limit, keyword count, exact template class, anonymization rules, AI-use declaration wording, table caption position.
