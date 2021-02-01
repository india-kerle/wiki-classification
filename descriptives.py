import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def most_frequent_terms(text_data, top_terms):
    '''takes as input cleaned text data and outputs top x most common terms.'''
    
    words = [y for x in text_data.tolist() for y in x.split()]

    # generate DF out of Counter
    rslt = pd.DataFrame(Counter(words).most_common(top_terms),
                        columns=['Word', 'Frequency'])
    
    return(rslt)

def condition_counts(text_data, cond1 = 1, cond2 = 0):
	'''takes as input comment label as toxic or obscene (1, 0) and outputs graph 
	showcasing the raw counts of comments labeled toxic, obscene, both, or neither.'''

	data = [["toxic", len(text_data[text_data.toxic == cond1])], ["obscene", len(text_data[text_data.obscene == cond1])], ["both", len(text_data[(text_data.toxic == cond1) & (text_data.obscene == cond1)])], ["neither", len(text_data[(text_data.toxic == cond2) & (text_data.obscene == cond2)])]]

	summary_stats = pd.DataFrame(data, columns = ["Condition", "Count"]).set_index('Condition')
	summary_stats.plot(kind = "bar")
	plt.ylabel('Count')

	return plt.show() 

def new_features(data):
	'''takes as input data and creates additional feature columns: total length, number of capiitals, caps vs. length, number of exclamation mark, number of punctuation and number of unique words.'''

	data['total_length'] = data['comment_text'].apply(len)
	data['capitals'] = data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
	data['caps_vs_length'] = data.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
	data['num_exclamation_marks'] = data['comment_text'].apply(lambda comment: comment.count('!'))
	data['num_punctuation'] = data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
	data['num_unique_words'] = data['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))

	return data