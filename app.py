from flask import Flask, render_template, request, redirect, url_for
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import difflib
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
#/Users/nandikaprince/Downloads/cleaned_books_data .csv
app = Flask(__name__)

#  VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

cleaned_books_data = pd.read_csv('/Users/nandikaprince/Downloads/cleaned_books_data .csv')

cleaned_books_data['year'] = cleaned_books_data['year'].astype(str)

# Combine features into a single column 
combined_features = cleaned_books_data['desc'] + ' ' + cleaned_books_data['authors'] + ' ' + cleaned_books_data['categories'] + ' ' + cleaned_books_data['year']

# Vectorize the combined features
vectorizer = TfidfVectorizer(stop_words='english')
desc_features = vectorizer.fit_transform(cleaned_books_data['desc'])
categories_features = vectorizer.fit_transform(cleaned_books_data['categories'])
authors_features = vectorizer.fit_transform(cleaned_books_data['authors'])
year_features = vectorizer.fit_transform(cleaned_books_data['year'])

desc_weight = 2.0
categories_weight = 1.8
authors_weight = 1.2
year_weight = 1.0

# Apply weights
weighted_desc_features = desc_features * desc_weight
weighted_categories_features = categories_features * categories_weight
weighted_authors_features = authors_features * authors_weight
weighted_year_features = year_features * year_weight

# Normalize weighted features
normalized_desc_features = normalize(weighted_desc_features)
normalized_categories_features = normalize(weighted_categories_features)
normalized_authors_features = normalize(weighted_authors_features)
normalized_year_features = normalize(weighted_year_features)

# Combine features using horizontal stacking
weighted_combined_features = hstack([
    normalized_desc_features,
    normalized_categories_features,
    normalized_authors_features,
    normalized_year_features
])

# Compute similarity
similarity = cosine_similarity(weighted_combined_features)
list_of_titles = cleaned_books_data['title'].tolist()

# get sentiment score using VADER
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']


cleaned_books_data['sentiment_score'] = cleaned_books_data['desc'].apply(get_sentiment_score)

def get_recommendations(fav_book, sentiment_filter, top_n=10):
    find_close_match = difflib.get_close_matches(fav_book, list_of_titles)
    closest_match = find_close_match[0]
    match_score = difflib.SequenceMatcher(None, fav_book, closest_match).ratio()
    
        # Define a threshold for similarity (e.g., 0.8)
    similarity_threshold = 0.8
    if len(find_close_match) == 0 or match_score < similarity_threshold:
        return [], False
    else:
        close_match = find_close_match[0]
        index_of_book = cleaned_books_data[cleaned_books_data.title == close_match].index.values[0]
        similarity_score = list(enumerate(similarity[index_of_book]))
        sorted_similar_books = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        # Filter recommendations by sentiment
        filtered_recommendations = []
        seen_books = set()
        for book in sorted_similar_books:
            index = book[0]
            if len(filtered_recommendations) >= top_n:
                break
            book_title = cleaned_books_data.iloc[index]['title']
            if book_title in seen_books:
                continue
            sentiment_score = cleaned_books_data.loc[index, 'sentiment_score']
            if (sentiment_filter == 'feel-good' and sentiment_score > 0.2) or (sentiment_filter == 'emotional' and sentiment_score < -0.2):
                book_info = cleaned_books_data.iloc[index]
                book_info['authors'] = ', '.join(eval(book_info['authors']))  # Convert authors list to a string
                if book_info['title'] != fav_book:
                    filtered_recommendations.append(book_info)
                    seen_books.add(book_title)
        return filtered_recommendations, True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        fav_book = request.form['book_title']
        sentiment_filter = request.form['sentiment_filter']
        recommendations, book_found = get_recommendations(fav_book, sentiment_filter)
        if not book_found:
            return render_template('error.html', fav_book=fav_book)
        return render_template('recommendations.html', fav_book=fav_book, recommendations=recommendations)
    return render_template('index.html')

@app.route('/home')
def home():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
