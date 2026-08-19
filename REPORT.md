# Reproduction report

I reimplemented the paper's XGBoost agent attribution pipeline and verified it against the authors' released traces and classifier artifacts. Percentages are macro F1 unless noted.

## 1. Reproduction vs published numbers

The pipeline reproduces the paper to within ~1pp everywhere I could check. That lets me attribute my negative results (in later sections) to the method itself as opposed to my implementation. 

In-domain 14-way closed set (paper Table 9):

| dataset | paper | this repo | delta (pp) | with their params |
|---|---|---|---|---|
| 2WikiMultiHop | 79.36 | 80.53 | +1.17 | 80.72 |
| FRAMES | 75.27 | 76.16 | +0.89 | 76.16 |
| WebShop | 74.23 | 75.06 | +0.83 | 75.06 |
| DeepShop | 72.57 | 70.18 | -2.39 | 73.29 |

DeepShop is the only noticeable gap, and it comes from search variance, not the feature pipeline. Their published parameters bring it to 73.29 (a 0.72pp difference).

Pooled-site and cross-site (paper Table 14):

| setting | paper | this repo |
|---|---|---|
| Wikipedia pooled, 2WikiMultiHop test | 81.3 | 81.68 |
| Wikipedia pooled, FRAMES test | 77.2 | 76.04 |
| Amazon pooled, WebShop test | 78.8 | 79.80 |
| Amazon pooled, DeepShop test | 70.9 | 72.52 |
| Wikipedia pooled -> Amazon | 29.70 | 29.86 |
| Amazon pooled -> Wikipedia | 25.98 | 27.67 |

The open-set results also line up well. All 56 unseen agent folds match  the per fold AUROC in Figure 3 (mean absolute difference 0.018, max 0.097), and known/unknown counts match in every fold. Seed-2.0-Lite is the strongest closed-set agent, but when treated as an unknown agent, it falls below the chance on 3/4 benchmarks. 

## 2. Data splits and feature verification

There’s a mismatch between the published FRAMES split record and the split used to train the released models.

- The split record keeps any trace with a non-empty event list, while the training code also removes API error traces. That changes 14 traces, and since the split routine uses one shared RNG across agents, those 14 also change later shuffles.
- I can reproduce both versions exactly, suggesting a documentation issue.  The training filter gives the same split sizes as the released models, and all 56 fold counts match.

I also checked the feature extractor on 2,000 traces. 39 of the 41 features match the reference implementation exactly (results/feature_validation.csv). The two differences come from following the Table 8 description, with neither one changing the results above.

## 3. Metric choice on a single confusion matrix

The headline 96% is the best performing agent out of 14. The same predictions result in very different numbers depending on what's reported (results/metrics_comparison.csv).

| dataset | macro F1 | best agent | worst agent | chance | permuted macro F1 |
|---|---|---|---|---|---|
| 2WikiMultiHop | 80.53 | 96.00 (Seed-2.0-Lite) | 67.95 (GLM-4.6V) | 7.14 | 7.12 +/- 0.81 |
| FRAMES | 76.16 | 97.40 (Qwen3.5-27B) | 50.00 (Gemini-3-Flash) | 7.14 | 7.11 +/- 0.79 |
| WebShop | 75.06 | 94.04 (Seed-2.0-Lite) | 46.77 (GLM-4.6V-Flash) | 7.14 | 7.08 +/- 0.82 |
| DeepShop | 70.18 | 88.61 (Seed-2.0-Lite) | 53.16 (Gemma-4-31B) | 7.14 | 7.09 +/- 1.15 |

Across all 14 agents, macro F1 tops out at 80.5, while the worst agent is at 46.8. So when comparing this result to other attribution papers, the metric and chance baseline are significant. 

## 4. Open-set evaluation with error-rate metrics

An AUROC around 0.7 seems workable until I use a false alert rate that would actually be reasonable in practice., At that point, detection is almost non-functional.

| group | folds | AUROC | EER | OSCR | miss @ 1% FAR | miss @ 5% FAR |
|---|---|---|---|---|---|---|
| easy | 16 | 63.7 | 40.2 | 54.9 | 97.6 | 90.0 |
| hard | 40 | 69.6 | 36.4 | 59.8 | 97.0 | 87.1 |

- At a 1% false alert rate, 97% of traces from an unseen agent classify as known. At 5%, 87% pass.
- This follows  the paper's unseen agent setup (results/open_set.csv). Each fold trains on 13 agents and uses the max class probability as the unknowness score. The easy folds are the ones where the held out agent has no sibling lineage in training.

I also expected agents from families represented in the training set to be harder to detect, but the results don’t show a consistent family effect. The overall gap is 5.9pp in the opposite direction, mostly because of Seed-2.0-Lite. Removing it gives 68.5 for easy folds versus 69.6 for hard folds, and using the family definitions from Table 11 reduces the gap to 2.6pp. In practice, which agent gets held out seems to matter more than if its family is represented.

## 5. Cross-site attribution and training-site scaling

Adding more environments does not make the fingerprint site independent. Most of the signal still seems to be related to the site where the traces were collected.

| k | raw | fixed budget | sibling in | sibling out |
|---|---|---|---|---|
| 1 | 28.6 | 27.2 | 47.8 | 20.3 |
| 2 | 39.4 | 34.3 | 43.1 | 25.6 |
| 3 | 45.3 | 37.3 | 40.5 | 27.6 |
| 4 | 49.6 | 39.6 | 39.6 | (none) |

- The raw numbers go up because the training set gets larger as more environments are added. In the runs, every mix is subsampled to 1,023 traces, so adding an environment doesn’t also mean adding more data.
- At a fixed budget, a held out site with no same domain sibling improves only from 20.3 to 27.6. Sibling in drops from 47.8 to 39.6, since the site carrying most of the signal makes up a smaller share of the training data.
- The sibling free k = 3 point appears because I added WebGames, the fifth environment in the release. It transfers at 11.7 on its own, which is barely above the chance level. 

My takeaway is that there seems to be some transferable signal, but it's underneath a much stronger site specific signal. It seems it's not pure shortcut learning or a model level fingerprint. For an actual deployment, the results suggest you’d want traces from the site where the detector will be used. That conclusion is based on three sibling free points across two domains and one game environment, collected through the same environment. 

## 6. Not reproduced and known limitations

- I didn't reproduce random forest, logistic regression, LSTM results, delay injection attack, family level classification, the sample efficiency, or early identification analyses.
- I ran the hyperparameter search once at seed 42 to match the paper. Only section 5 uses multiple seeds, so DeepShop is the best indication I have of single search variance.
- I kept the reference implementation's quirks for comparability. So, these results are based on one harness, one provider's model lineup, and five environments.

## 7. Extension attempt: harness decay across a four-month gap

I tried collecting WebShop traces for newer Claude versions using the authors' harness, four months after their original collection window. Even though it no longer works, here are the results I found:

- Opus 5 finished 0/5 episodes within the paper's 300s limit, making around 60 model calls per attempt. Opus 4.7 finished 3 of 3 in 137–201s.
- Opus 4.7, 4.8 and the Claude 5 models reject the temperature parameter used in the April collection (400, deprecated), so those runs fell back to vendor defaults instead.
- The current MidScene version won't accept the family omitted config used for the original Claude traces, so I needed an older version plus a planning model config that wasn't included in the release.
- WebShop runs on live amazon.com, which now serves bot detection interstitials. Sonnet 5 completed its first episode but failed the next eight.

In total, I got 6 usable traces for $48.88 and no usable bridge episodes. This goes to show how quickly models are changing and how harnesses & their results become deprecated, even in just in four months!