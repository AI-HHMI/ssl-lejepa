The driving question is: is LeJEPA a better training scheme than MAE for a microscopy foundation model?

To do that we have to optimize LeJEPA training.

The key areas for optimization are

1. view creation
2. architecture.

We ran experiments

- e00/viewsizes-pca
- e00/viewsizes-v2
- e00/viewsizes_wd
- e00/viewsizes

to explore the effects of view creation params 

- `patch_size`
- `global_size`
- `local_size`.

The analysis studied

- PCA maps
