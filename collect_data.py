"""
collect_data.py

Gathers and cleans data from Brewery DB.
"""

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


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


def pale_ale(api_key):
    """Create table of American pale ales."""
    r = requests.get(base + 'beers/?styleId=25&withBreweries=Y&key=%s' %
                     api_key).json()
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
                 'name', 'style.category.name']
    df = df[keep_cols]

    # Write to output
    df.to_csv('pale.txt', sep='|', index=False, encoding='utf-8')


def tf_idf(df):
    """Creates matrix of tf-idf scores. Returns sparse tf-idf matrix
    and ordered list of words used in the matrix as a list.

    df: Data frame with a 'description' column used for the tf-idf
    """
    tf = TfidfVectorizer()
    tf_idf_mat = tf.fit_transform(df['description'])

    return [tf_idf_mat, tf.get_feature_names()]


def high_scoring_features(row, features, n=5):
    """Returns the n highest scoring words in a document (row)
    from the tf-idf score matrix.

    row: array of tf-idf scores for each word in a document
    features: list of features with same index mapping used in tf-idf
    """
    # Get indexes of top scoring features
    top_idx = np.argsort(row)[::-1][:n]

    # Create data frame of most important words and tf-idf score
    top_feats = [(features[i], row[i]) for i in top_idx]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf_score']
    return df


if __name__ == '__main__':

    # API configuration
    with open('settings.config', 'r') as config:
        api_key = config.readline().strip()
    base = 'http://api.brewerydb.com/v2/'

    # Read/save first page of data
    pale_ale(api_key)

    # Read data frame, create tf-idf matrix
    pale = pd.read_csv('pale.txt', sep='|')
    m = tf_idf(pale)
    document = m[0][0,:].toarray()[0]
    feature_list = m[1]

    # Check most important features in first document
    top_features = high_scoring_features(document, feature_list)
