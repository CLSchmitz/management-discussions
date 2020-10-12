import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, stem_text, strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short
from gensim.test.utils import get_tmpfile

#Reading in Data
print('reading data')
deltas_df = pd.read_csv('mgmt_sections_and_compustat_incl_delta_4.csv')

#Reformatting from every company having its own row to every report having its own row
deltas_df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis = 1, inplace = True)
rf_df = pd.wide_to_long(deltas_df,
                        stubnames = ['report_',
                                     'date_',
                                     'total_comp_income_',
                                     'dividends_', 
                                     'revenue_', 
                                     'stockholders_equity_', 
                                     'market_value_',
                                     'delta_comprehensive_income_',
                                     'delta_dividends_',
                                     'delta_revenue_',
                                     'delta_stockholders_equity_',
                                     'delta_market_value_',
                                     'delta_market_value_forward_'],
                        i = 'company_cik',
                        j = 'temp_index').reset_index(drop = True)

#Dropping rows without reports
missing_indexes = rf_df[rf_df['report_'] == 'not_found'].index
rf_df.drop(missing_indexes, inplace = True)

reports = rf_df['report_']

#Cleaning and tokenizing reports
print('cleaning reports')
CUSTOM_FILTERS = [lambda x: x.lower(),
                  strip_multiple_whitespaces,
                  strip_non_alphanum,
                  strip_numeric,
                  remove_stopwords,
                  strip_short,
                  stem_text]

processed_reports = [preprocess_string(x, CUSTOM_FILTERS) for x in reports] #this line takes long
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_reports)]

#Training model
print('training model')
model = Doc2Vec(documents, vector_size=30, window=3, min_count=1, workers=4) #this line takes long

#Saving model just in case
fname = get_tmpfile("mgmt_disc_model")
model.save(fname)

#Creating report vectors
report_vectors = []
for report in processed_reports:
    report_vectors.append(model.infer_vector(report))

#Creating df of report vectors and naming cols
report_vectors_df = pd.DataFrame.from_records(report_vectors)
col_names = ["doc2vec_dim_{}".format(i+1) for i in range(30)]
report_vectors_df.columns = col_names

#Concatinating dataframes
rf_df = rf_df.reset_index(drop = True)
final_df = pd.concat([rf_df, report_vectors_df], axis = 1)

#Dropping raw reports
final_df.drop(['report_'], axis = 1, inplace = True)

#Saving DF
final_df.to_csv('clean_no_drops.csv')
