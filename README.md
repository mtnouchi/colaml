# colaml
CoLaML is a tool for joint inference of ancestral gene **Co**ntents and **La**tent evolutionary modes by **M**aximum **L**ikelihood method

## Requirements

See `pyproject.toml`.

## Installation

To install CoLaML, run

```
pip install .
```

This will also install its CLI `colaml`.

## Usage - CLI

### Input File Format

To run CoLaML, you need to prepare a input json file describing phylogenetic tree and ortholog table:

- `tree` field: Phylogenetic tree in the Newick format string. Recognized as [format 3 in ete3](http://etetoolkit.org/docs/latest/reference/reference_tree.html). Tree leaves must be uniquely named.

- `OGs` field: ortholog table
  - `index`: name of tree leaves as a list of strings
  - `columns`: (optional) name of orthologs as a list of strings
  - `data`: 2D-array of gene copy numbers in the order specified by `index` and `columns`. Values are automatically clipped to the range `[0, lmax]`. 


### Model Fitting

Three models are available:

- `mmm`   : Markov-modulated model (our proposed model)
- `mirage`: Mirage: mixture model ([REF](https://doi.org/10.1093/bioadv/vbab014))
- `branch`: Branch model


```
$ colaml fit mmm --help

usage: colaml fit mmm [-h] [-q] -i INPUT -o OUTPUT [--max-iter MAX_ROUNDS]
                      --seed SEED --lmax LMAX --ncat NCAT [--map]
                      [--gainloss-gamma SHAPE,SCALE]
                      [--switch-gamma SHAPE,SCALE]
                      [--copy-root-dirichlet ALPHA[,ALPHA[,...]]]
                      [--cat-root-dirichlet ALPHA[,ALPHA[,...]]]

options:
  -h, --help            show this help message and exit
  -q, --no-progress     suppress progress bar
  -i INPUT, --input INPUT
                        path to input json file (can be gzipped)
  -o OUTPUT, --output OUTPUT
                        path+prefix for output files
  --max-iter MAX_ROUNDS
                        maximum iterations in EM
  --seed SEED           random seed
  --lmax LMAX           max gene copy number
  --ncat NCAT           #(rate categories)

advanced options for MAP estimation:
  --map                 enable MAP estimation
  --gainloss-gamma SHAPE,SCALE
                        gamma prior of gain/loss rate
  --switch-gamma SHAPE,SCALE
                        gamma prior of switch rate
  --copy-root-dirichlet ALPHA[,ALPHA[,...]]
                        Dirichlet prior of copy root probs
  --cat-root-dirichlet ALPHA[,ALPHA[,...]]
                        Dirichlet prior of category root probs
```

Usage of the other two models can be examined in the same way with `--help` option.

If finer control is needed, use the Python package.

### Ancestral State Reconstruction

```
$ colaml recon --help

usage: colaml recon [-h] -i INPUT -m MODEL -o OUTPUT --method {joint,marginal}

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input json file (can be gzipped)
  -m MODEL, --model MODEL
                        path to fitting json file (can be gzipped)
  -o OUTPUT, --output OUTPUT
                        path to output file
  --method {joint,marginal}
                        reconstruction method
```

## Usage - Package

(Under construction)

## Test datasets

See [here](https://github.com/mtnouchi/colaml-test).

## Citation

(TBA)
