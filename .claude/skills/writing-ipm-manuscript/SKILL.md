---
name: writing-ipm-manuscript
description: Use when drafting, rewriting, proofreading or reformatting the MPUP paper (paper/mpup.tex, abstract, highlights, section structure, captions, end-matter declarations) for submission to Information Processing & Management (IP&M, Elsevier), or converting it from the IEEEtran conference layout.
---

# Writing for Information Processing & Management

Rules below come from six IP&M papers (accepted versions, 2025-2026) read in full; evidence is in `corpus-notes.md`. The official Guide for Authors could not be fetched (ScienceDirect blocks automated access). Items marked **[verify]** come from Elsevier's general policy, not from the corpus. Check them at the Guide for Authors before submitting.

## Format

| Element | IP&M convention |
|---|---|
| Template | Elsevier `elsarticle` / `cas-sc`, single column, author-year citations, reference list sorted by author **[verify class]** |
| Highlights | 3-5 bullets, each at most 85 characters, separate file **[verify]**. Each states a result or contribution, not a topic. |
| Abstract | One paragraph, 190-320 words in the corpus (most 240-280); no limit confirmed **[verify]**. Order: gap, method, data and setup, headline numbers, implication. |
| Keywords | 5-6, noun phrases, lowercase except acronyms |
| Review | Double-anonymized: title page with authors in a separate file, no names, funding or self-identifying repo links in the manuscript **[verify]** |
| Length | 20-44 single-column pages in the corpus; no tight limit observed |
| Captions | `Figure N: Sentence-case description.` Figure captions below, table captions above. Say what the reader should conclude. |
| End matter, in order | Acknowledgements, CRediT authorship contribution statement, Declaration of competing interest, Data availability (with code URL), generative-AI use declaration **[verify]**, References |

## Section skeleton

1. Introduction (task, limits of existing work, what we propose, numbered contributions)
2. Related work (subsections by approach family, ending with the research gap)
3. Research objectives or research questions (RQ1..RQn, 2 of 6 papers; fits MPUP's C1-C4 criteria)
4. Methodology
5. Experimental setup (datasets, baselines, metrics, implementation)
6. Results, one subsection per RQ, each closing with a short boxed "RQn summary answer"
7. Ablation / analysis
8. Discussion: theoretical contributions, **practical or information-management implications**, limitations, scope and external validity
9. Conclusion (restate the headline numbers, then future work)
Appendix after the conclusion.

IP&M reviewers expect the implications subsection. Frame the work as information selection, retrieval or management, as SegRAG does.

## Language

- Active voice, "we" or "this paper". Present tense for the method and general claims, past tense for what was run ("results showed").
- Quantify. Abstract and conclusion carry the same headline numbers (effect sizes, deltas, test statistics).
- Contributions as one sentence: "In summary, our contributions are: (1) ...; (2) ...; (3) ...".
- Define each acronym at first use. Use one term per concept (utility, not usefulness/helpfulness interchangeably).
- Calibrated verbs: "suggest", "indicate", "demonstrate" only when a test supports it. State scope: "The experiments establish X for Y; separate evaluation is needed for Z."
- No slogans or colon titles. Descriptive, problem-first title (see `paper/title-options.md`).
- Every sentence needs a grammatical subject and verb. Check for the pattern "X making them cannot Y" and for "methods proven to be" (missing auxiliary).

## Converting this repo's paper from IEEEtran

1. Replace `IEEEtran` class, `IEEEkeywords`, `\cite` numeric style with the Elsevier class and author-year bibliography.
2. Split "Experimental Setup" (currently holds results, limitations, cost) into setup, results per criterion, discussion, limitations.
3. Add highlights, research questions, implications subsection, end-matter declarations.
4. Remove `\TODO` markers; anonymize for review.
5. Rebuild `paper/mpup.pdf` and read it before committing.

## Common mistakes

- Treating the Guide-for-authors **[verify]** items as confirmed.
- Copying sentences from the corpus papers. Use them for structure and register only.
- Highlights that restate the abstract's first sentence instead of results.
- Putting limitations only inside the setup section; IP&M papers discuss them in the Discussion.
