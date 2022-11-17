import pandas as pd
import pandera as pa
import sys
import typer

from collections import deque
from loguru import logger
from multiprocessing import Pool, cpu_count
from pandera.typing import DataFrame, Series, Object
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
from pathlib import Path


tqdm.pandas()

app = typer.Typer()

DATA_DIR = Path(__file__).parent / "data"
MALE_PATH = DATA_DIR / "male.csv"
FEMALE_PATH = DATA_DIR / "female.csv"


class Wikidata(pa.SchemaModel):

    wikidata_id: Series[pd.StringDtype] = pa.Field(unique=True)
    results: Series[Object] = pa.Field(nullable=True)

    @pa.check("wikidata_id")
    def check_wikidata_id(cls, col):
        return col.str.startswith("Q")


@app.command()
def male():
    query = """
    SELECT ?human
    WHERE {
        ?human wdt:P31 wd:Q5;
        wdt:P21/wdt:P279* wd:Q6581097
    }
    LIMIT 100000
    """

    results = get_results(query)
    df = pd.json_normalize(results)
    df.columns = ["type", "wikidata"]
    df = df.drop("type", axis=1)
    df["wikidata_id"] = df.wikidata.progress_apply(lambda x: x.split("/")[-1])
    df.to_csv(MALE_PATH, index=False)


@app.command()
def female():
    query = """
    SELECT ?human
    WHERE {
        ?human wdt:P31 wd:Q5;
        wdt:P21/wdt:P279* wd:Q6581072
    }
    LIMIT 100000
    """

    results = get_results(query)
    df = pd.json_normalize(results)
    df.columns = ["type", "wikidata"]
    df = df.drop("type", axis=1)
    df["wikidata_id"] = df.wikidata.progress_apply(lambda x: x.split("/")[-1])
    df.to_csv(FEMALE_PATH, index=False)


@app.command()
def results(path):
    df = pd.read_csv(path)
    df = df[["wikidata_id"]]
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df["query"] = df["wikidata_id"].progress_apply(get_query)

    n_cores = cpu_count() // 4
    logger.info(f"Collecting Wikidata with {n_cores} processes")
    with Pool(n_cores) as pool:
        results = pool.imap(get_results, df["query"])
        df["results"] = deque(tqdm(results, total=len(df["wikidata_id"])))

    df = df[["wikidata_id", "results"]]
    df = df.convert_dtypes()
    df = df.dropna()
    df = df.reset_index(drop=True)

    df.to_csv(f"{path}".replace(".csv", "-results.csv"), index=False)


@app.command()
def build(path):
    pass


def get_query(wikidata_id):
    template = """
    SELECT
        ?item
        ?itemLabel
    WHERE {
        VALUES ?item { wd:%s }
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        }
    }
    """

    if wikidata_id:
        return template % (wikidata_id)


def get_agent():
    info = sys.version_info
    return f"Wikigender. Python/{info[0]}.{info[1]}"


def get_results(query, api="https://query.wikidata.org/sparql"):
    """
    TODO: Implement retries for JSONDecodeError.
    `JSONDecodeError: Invalid control character at: line _ column _ (char _)`
    """
    agent = get_agent()

    sparql = SPARQLWrapper(api, agent=agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.warning(e)


if __name__ == "__main__":
    app()
