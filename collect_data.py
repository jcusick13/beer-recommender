"""
collect_data.py

Gathers and cleans data from Brewery DB.
"""

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Create and save table of beer styles
# r = requests.get('http://api.brewerydb.com/v2/styles?key=%s' % api_key).json()

# styles = pd.DataFrame.from_dict(json_normalize(r['data']), orient='columns')

# drop_cols = [u'category.createDate', u'category.description',
#              u'category.updateDate', u'createDate', u'ogMax',
#              u'updateDate']
# styles.drop(drop_cols, axis=1, inplace=True)
# styles.to_csv('styles.txt', sep=',', index=False, encoding='utf-8')


def test_glassware():
    r = requests.get(base + 'glassware?key=%s' % api_key).json()
    df = pd.DataFrame.from_dict(json_normalize(r['data']), orient='columns')
    df.to_csv('test.txt', sep='|', index=False, encoding='utf-8')


def download_style_data(api_key, style_id, outfile):
    """Creates table from first returned page for a given style of beer.

    api_key (str): Key for api access
    style_id (int): Added to API request for a specific style of beer
    outfile (str): Name of output txt to write data to
    """
    r = requests.get(base + 'beers/?styleId=%s&withBreweries=Y&key=%s' %
                     (style_id, api_key)).json()
    df = pd.DataFrame.from_dict(json_normalize(r['data']), orient='columns')

    # Clean up characters in order to output as csv
    df.replace('|', ':', inplace=True)
    df.replace('"', '', inplace=True)
    df['description'] = df.description.str.replace(r'\r', '')
    df['description'] = df.description.str.replace(r'\n', ' ')

    # Select brewery name from nested json, merge into original data frame
    def _get_name(x):
        return x[0]['nameShortDisplay']
    names = pd.Series(df['breweries'].apply(_get_name))
    names = names.rename('brewer')
    df = pd.concat((df, names), axis=1, join='inner')

    # Remove beers with missing information, subset to relevant columns
    check_empty_cols = ['abv', 'ibu', 'description']
    df.dropna(axis=0, how='any', subset=check_empty_cols, inplace=True)
    df = df.loc[df['available.id'] != 3]  # Beer not available
    keep_cols = ['abv', 'brewer', 'description', 'ibu', 'id', 'isOrganic',
                 'name', 'style.shortName']
    df = df[keep_cols]

    # Write to output
    df.to_csv(outfile, sep='|', index=False, encoding='utf-8')


def tf_idf(df):
    """Creates matrix of tf-idf scores. Returns sparse tf-idf matrix
    and ordered list of words used in the matrix as a list.

    df: Data frame with a 'description' column used for the tf-idf
    """
    tf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tf_idf_mat = tf.fit_transform(df['description'])

    return [tf_idf_mat, tf.get_feature_names()]


def high_scoring_features(row, features, n=5):
    """Returns the n highest scoring words in a document (row)
    from the tf-idf score matrix.

    row: array of tf-idf scores for each word in a document
    features: list of features with same index mapping used in tf-idf
    """
    # Get indexes of top scoring features
    top_idx = np.argsort(-row)[:n]

    # Create data frame of most important words and tf-idf score
    top_feats = [(features[i], row[i]) for i in top_idx]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf_score']
    return df


def similar_documents(idx, tfidf_mat, orig_df, feat_list, n=3):
    """Computes the cosine similarity between doc and each entry
    in the tfidf matrix. Returns the top n results.

    idx: int row number of item in tfidf_mat to find similar items for
    tfidf_mat: sparse matrix of tf-idf scores for corpus of documents
    orig_df: original dataframe used to build tfidf_mat
    feat_list: list of all features (words) in document corpus
    n: number of top matching results to return
    """

    # Select input matrix vals, reshape to array, compute similiarity score
    doc = tfidf_mat[idx, :].toarray()
    cos_res = cosine_similarity(doc, tfidf_mat)

    # Sort and select top n results; ignore the highest result. It will
    # always be the document (doc) itself, with a score of 1 since it is
    # identical to itself.
    top_idx = np.argsort(-cos_res)[0][1:n+1]

    # Sanity check results
    print('\n** Similar to %s (%s) **' %
          (orig_df['name'][idx], orig_df['style.shortName'][idx]))
    print('High Scoring Features:')
    print(high_scoring_features(doc[0], feat_list, n=30))
    for i in top_idx:
        print('\n%s (%s) Description:' %
              (orig_df['name'][i], orig_df['style.shortName'][i]))
        print(orig_df['description'][i])

    return orig_df.iloc[top_idx, :]


if __name__ == '__main__':

    # API configuration
    with open('settings.config', 'r') as config:
        api_key = config.readline().strip()
    base = 'http://api.brewerydb.com/v2/'

    # Read/save first page of data
    #download_style_data(api_key, 18, 'brown_porter.txt')

    # Download sample pale ale and porter data/combine
    pale = pd.read_csv('american_pale.txt', sep='|')
    porter = pd.read_csv('brown_porter.txt', sep='|')
    blonde = pd.read_csv('blonde.txt', sep='|')
    beers = porter.append(pale, ignore_index=True)
    beers = beers.append(blonde, ignore_index=True)

    # Create tf-idf matrix
    m = tf_idf(beers)
    feature_list = m[1]

    # Find similar documents
    most_similar = similar_documents(20, m[0], beers, feature_list)

    # TODO: Create function to find ngrams most similar bewteen two documents
    # TODO: Expand data collection to all pages of beers per style
    # TODO: Combine text filtering with numerical attributes
    # TODO: Break into separate files for collection/analysis

