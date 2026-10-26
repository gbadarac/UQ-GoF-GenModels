# Uncertainty-aware validation of generative models: paper code

Code for *An end-to-end framework for uncertainty-aware validation of generative models*
(G. Badaracco, S. Benevedes, C. Reissel, P. Harris, G. Grosso).

| Directory | Contents | Paper |
|---|---|---|
| [`sigma/`](sigma/) | SIGMA: Gaussian-mixture density estimation, tangent-space uncertainties, the composite goodness-of-fit test with bootstrap, Gaussian and point nulls, and the exported data and scripts for every GMM figure | Secs. IV–V, Apps. C–E; Figs. 4, 5, 7, 8, 9, 14, 15, 16 |
| [`nf_wifi/`](nf_wifi/) | Ensembles of normalizing flows with w_i f_i weight uncertainties, and the same test with point and Gaussian nulls | App. B; Figs. 10–13 |

The two parts are independent, with their own environments and READMEs. They share the 2D toy distribution (the same
generator, seed 42) and the 4D JetClass QCD embedding, which is not included here. Each README describes how that part
splits the 2M-jet pool.

Figs. 1–3 and 6 are schematics or one-off illustrations and are not produced by this code.

To redraw the GMM figures without a cluster: `bash sigma/make_figures.sh`.
