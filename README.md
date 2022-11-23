# wikigender

Dataset of names and gender from Wikidata

1. Scrape male and female wikidata items

> python wikigender.py male

> python wikigender.py female

2. Scrape name of each set of items

> python wikigender.py results data/male.csv

> python wikigender.py results data/female.csv

3. Parse results from each set of results

> python wikigender.py parse data/male-results.csv

> python wikigender.py parse data/female-results.csv

4. Merge the parsed results to build the dataset

> python wikigender.py build data/male-results-parsed.csv data/female-results-parsed.csv

