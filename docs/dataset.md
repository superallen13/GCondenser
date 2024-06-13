# Supported dataset
For `GraphCondenser`, budget is the node number of the condensed graph. In most of graph condensation paper, condense rate (e.g., the ratio of the size of the condensed graph to the number of nodes that can be observed during the condensing process.) are reported.

| Dataset | $N_{cls}$ | $N_{\text{train}}$ | $N_{full}$ | $r_{\text{train}}$ | $r_{\text{full}}$ | $N_{cond}$ |
|-------|-------|-------|-------|-------|-------|-------|
| Citeseer | 6 | 120 | 3,327 | 25% <br> 50% <br> 100% | 0.9% <br> 1.8% <br> 3.6% | 30 <br> 60 <br> 120 |
| Cora | 7 | 140 | 2,708 | 25% <br> 50% <br> 100% | 1.3% <br> 2.6% <br> 5.2% | 35 <br> 70 <br> 140 |
| Pubmed | 3 | 60 | 19,717 | 25% <br> 50% <br> 100% | 0.08% <br> 0.15% <br> 0.3% | 15 <br> 30 <br> 60 |
| ogbn-arxiv | 40 | 90,941 | 169,343 | 0.1% <br> 0.5% <br> 1% | 0.05% <br> 0.25% <br> 0.5% | 90 <br> 454 <br> 909 |
| ogbn-products | 47 | 196,615 | 2,449,029 | 0.3% <br> 0.6% <br> 1.2% | 0.025% <br> 0.05% <br> 0.1% | 612 <br> 1,225 <br> 2,449 |
| Flickr | 7 | 44,625 | 89,250 | 0.1% <br> 0.5% <br> 1% | 0.1% <br> 0.5% <br> 1% | 44 <br> 223 <br> 446 |
| Reddit | 41 | 15,3932 | 23,2965 | 0.1% <br> 0.5% <br> 1% | 0.05% <br> 0.1% <br> 0.2% | 153 <br> 769 <br> 1,539 |