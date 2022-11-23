# wikigender

Dataset of names and gender from Wikidata


## Setup

Install `poetry` from https://python-poetry.org/docs/#installation

```sh
poetry install
```

## Build

1. Scrape male and female wikidata items

```sh
python wikigender.py male
python wikigender.py female
```

2. Scrape name of each set of items

```sh
python wikigender.py results data/male.csv
python wikigender.py results data/female.csv
```

3. Parse results from each set of results

```sh
python wikigender.py parse data/male-results.csv
python wikigender.py parse data/female-results.csv
```

4. Merge the parsed results to build the dataset

```sh
python wikigender.py build data/male-results-parsed.csv data/female-results-parsed.csv
```
