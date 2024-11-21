from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, json
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast, AutoModel
import numpy as np
import torch.nn as nn
import plotly.express as px
import plotly.io as pio
import pycountry
import os

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define BERT architecture
class BERT_architecture(nn.Module):
    def __init__(self, bert):
        super(BERT_architecture, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the saved BERT model weights
model = BERT_architecture(bert)
model.load_state_dict(torch.load('saved_weights.pt'))
model.eval()

# Function to tokenize and encode sequences
def tokenize_sequences(texts, max_len=25):
    tokens = tokenizer.batch_encode_plus(
        texts,
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True
    )
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    return TensorDataset(seq, mask)

# Function to predict sentiment labels for tweets
def predict_sentiment(dataloader):
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            sent_id, mask = batch
            outputs = model(sent_id, mask)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
    return preds

# Global variable to store uploaded data
global_data = None

# Route to render the upload form for sentiment analysis
@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/sentiment', methods=['GET'])
def sentiment_index():
    return render_template('index1.html')

# Route to handle file upload and sentiment prediction
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_from_csv():
    global global_data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        df = pd.read_csv(file, engine="python", encoding='latin-1')
        df = df.sample(20)
        texts = df['text'].tolist()  # Adjust column name if needed
        dataloader = DataLoader(tokenize_sequences(texts), batch_size=64)
        predicted_sentiments = predict_sentiment(dataloader)
        sentiment_mapping = {0: 'negative', 1: 'positive'}
        df['predicted_sentiment'] = [sentiment_mapping[label] for label in predicted_sentiments]

        # Save the labeled dataset to a new CSV file
        labeled_csv_filename = 'labeled_tweets.csv'
        df.to_csv(labeled_csv_filename, index=False)

        # Store the data globally
        global_data = df

        return send_file(labeled_csv_filename, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)})

# Route to render the upload form for the world map and plotly plot
@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    global global_data
    if request.method == 'POST':
        try:
            # Get the uploaded CSV file
            csv_file = request.files['csv_file']

            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv(csv_file)
            global_data = df

            # Redirect to the table page
            return redirect(url_for('show_table'))

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('upload.html')

# Route to display the CSV data in a table
@app.route('/show_table', methods=['GET'])
def show_table():
    global global_data
    if global_data is None:
        return redirect(url_for('upload_csv'))

    data = global_data.to_dict(orient='records')
    columns = [{"title": col, "data": col} for col in global_data.columns]
    return render_template('table.html', data=json.dumps(data), columns=json.dumps(columns))

# Route to display the Plotly plot
@app.route('/show_plot', methods=['GET'])
def show_plot():
    global global_data
    if global_data is None:
        return redirect(url_for('upload_csv'))

    try:
        df = global_data

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'], format='%B %Y')

        # Create a pivot table for negative tweets
        pivot_table_negative = df[df['predicted_sentiment'] == 'negative'].pivot_table(index='date', columns='country', aggfunc='size', fill_value=0)
        top_countries_negative = pivot_table_negative.sum().nlargest(10).index
        pivot_table_negative = pivot_table_negative[top_countries_negative]
        melted_df_negative = pivot_table_negative.reset_index().melt(id_vars='date', var_name='Country', value_name='Tweet Count')

        # Create the Plotly figure for negative tweets
        fig_negative = px.line(melted_df_negative, x='date', y='Tweet Count', color='Country',
                               title='Top 10 Countries with Negative Tweets Over Time',
                               labels={'date': 'Time', 'Tweet Count': 'Number of Negative Tweets', 'Country': 'Country'},
                               template='plotly_white')
        fig_negative.update_layout(legend_title_text='Country', legend=dict(x=1.1, y=1), xaxis=dict(title='Time'), yaxis=dict(title='Number of Negative Tweets'))
        plot_negative_html = fig_negative.to_html(full_html=False)

        # Create a pivot table for positive tweets
        pivot_table_positive = df[df['predicted_sentiment'] == 'positive'].pivot_table(index='date', columns='country', aggfunc='size', fill_value=0)
        top_countries_positive = pivot_table_positive.sum().nlargest(10).index
        pivot_table_positive = pivot_table_positive[top_countries_positive]
        melted_df_positive = pivot_table_positive.reset_index().melt(id_vars='date', var_name='Country', value_name='Tweet Count')

        # Create the Plotly figure for positive tweets
        fig_positive = px.line(melted_df_positive, x='date', y='Tweet Count', color='Country',
                               title='Top 10 Countries with Positive Tweets Over Time',
                               labels={'date': 'Time', 'Tweet Count': 'Number of Positive Tweets', 'Country': 'Country'},
                               template='plotly_white')
        fig_positive.update_layout(legend_title_text='Country', legend=dict(x=1.1, y=1), xaxis=dict(title='Time'), yaxis=dict(title='Number of Positive Tweets'))
        plot_positive_html = fig_positive.to_html(full_html=False)

        return render_template('plot.html', negative_plot_html=plot_negative_html, positive_plot_html=plot_positive_html)

    except Exception as e:
        return jsonify({'error': str(e)})

# Route to display the world map
@app.route('/show_world_map', methods=['GET'])
def show_world_map():
    global global_data
    if global_data is None:
        return redirect(url_for('upload_csv'))

    try:
        df = global_data
        map_html = generate_map(df)
        return render_template('map.html', map_html=map_html)

    except Exception as e:
        return jsonify({'error': str(e)})

# Function to generate world map
def generate_map(df):
    # Group by country and predicted sentiment
    sentiment_counts = df.groupby(['country', 'predicted_sentiment']).size().unstack(fill_value=0)
    sentiment_counts['total'] = sentiment_counts.sum(axis=1)
    sentiment_counts['positive_pct'] = sentiment_counts.get('positive', 0) / sentiment_counts['total'] * 100
    sentiment_counts['negative_pct'] = sentiment_counts.get('negative', 0) / sentiment_counts['total'] * 100
    sentiment_counts['primary_sentiment'] = sentiment_counts.apply(
        lambda row: 'positive' if row.get('positive', 0) > row.get('negative', 0) else ('negative' if row.get('negative', 0) > 0 else 'neutral'), 
        axis=1
    )
    sentiment_counts = sentiment_counts.reset_index()



    

    country_to_iso = {
        'Armenia': 'ARM', 'Australia': 'AUS', 'New Zealand': 'NZL', 'United Kingdom': 'GBR',
        'South Korea': 'KOR', 'Turkey': 'TUR', 'Puerto Rico': 'PRI', 'Canada': 'CAN',
        'Vietnam': 'VNM', 'Greenland': 'GRL', 'Taiwan': 'TWN', 'Eritrea': 'ERI',
        'USA': 'USA', 'Oman': 'OMN', 'Russia': 'RUS', 'South Africa': 'ZAF',
        'Jordan': 'JOR', 'Nigeria': 'NGA', 'Germany': 'DEU', 'North Korea': 'PRK',
        'Israel': 'ISR', 'Maldives': 'MDV', 'China': 'CHN', 'Ethiopia': 'ETH',
        'Hong Kong': 'HKG', 'Tunisia': 'TUN', 'Iran': 'IRN', 'India': 'IND',
        'Ghana': 'GHA', 'Pakistan': 'PAK', 'Qatar': 'QAT', 'Austria': 'AUT',
        'Chile': 'CHL', 'Romania': 'ROU', 'Myanmar': 'MMR', 'Mexico': 'MEX',
        'Cambodia': 'KHM', 'Belarus': 'BLR', 'Ireland': 'IRL', 'Greece': 'GRC',
        'Azerbaijan': 'AZE', 'Latvia': 'LVA', 'Belgium': 'BEL', 'Brazil': 'BRA',
        'Afghanistan': 'AFG', 'Peru': 'PER', 'Yemen': 'YEM', 'Palestine': 'PSE',
        'Sri Lanka': 'LKA', 'Portugal': 'PRT', 'Bulgaria': 'BGR', 'Georgia': 'GEO',
        'Spain': 'ESP', 'Japan': 'JPN', 'Saudi Arabia': 'SAU', 'France': 'FRA',
        'Lebanon': 'LBN', 'El Salvador': 'SLV', 'Egypt': 'EGY', 'Italy': 'ITA',
        'Jamaica': 'JAM', 'Mozambique': 'MOZ', 'Kuwait': 'KWT', 'Cyprus': 'CYP',
        'Cameroon': 'CMR', 'Finland': 'FIN', 'Uganda': 'UGA', 'Albania': 'ALB',
        'Tanzania': 'TZA', 'Kyrgyzstan': 'KGZ', 'Croatia': 'HRV', 'Singapore': 'SGP',
        'Slovenia': 'SVN', 'Nepal': 'NPL', 'Switzerland': 'CHE', 'Indonesia': 'IDN',
        'Tajikistan': 'TJK', 'Poland': 'POL', 'Monaco': 'MCO', 'Malaysia': 'MYS',
        'Ukraine': 'UKR', 'Kenya': 'KEN', 'Djibouti': 'DJI', 'Colombia': 'COL',
        'Rwanda': 'RWA', 'Bahamas': 'BHS', 'Bangladesh': 'BGD', 'Denmark': 'DNK',
        'Bahrain': 'BHR', 'Estonia': 'EST', 'Luxembourg': 'LUX', 'Vanuatu': 'VUT',
        'Syria': 'SYR', 'Zambia': 'ZMB', 'Belize': 'BLZ', 'Iceland': 'ISL',
        'Bermuda': 'BMU', 'Iraq': 'IRQ', 'Thailand': 'THA', 'Bolivia': 'BOL',
        'Montenegro': 'MNE', 'Malta': 'MLT', 'Sweden': 'SWE', 'Uruguay': 'URY',
        'Morocco': 'MAR', 'Mauritius': 'MUS', 'Hungary': 'HUN', 'Guatemala': 'GTM',
        'Somalia': 'SOM', 'Norway': 'NOR', 'Czech Republic': 'CZE', 'Libya': 'LBY',
        'Algeria': 'DZA', 'Netherlands': 'NLD', 'Slovakia': 'SVK', 'Argentina': 'ARG',
        'Ecuador': 'ECU', 'Serbia': 'SRB', 'Mali': 'MLI', 'Zimbabwe': 'ZWE',
        'Seychelles': 'SYC', 'Tonga': 'TON', 'Isle of Man': 'IMN', 'Angola': 'AGO',
        'Gabon': 'GAB', 'Dominican Republic': 'DOM', 'Trinidad and Tobago': 'TTO',
        'New Caledonia': 'NCL', 'Costa Rica': 'CRI', 'Cuba': 'CUB', 'Chad': 'TCD',
        'Venezuela': 'VEN', 'Sudan': 'SDN', 'Senegal': 'SEN', 'Lithuania': 'LTU',
        'Congo': 'COG', 'Lesotho': 'LSO', 'Honduras': 'HND', 'Haiti': 'HTI',
        'Bosnia': 'BIH', 'Malawi': 'MWI', 'Antarctica': 'ATA', 'Kazakhstan': 'KAZ',
        'United Arab Emirates': 'ARE', 'Fiji': 'FJI', 'Namibia': 'NAM', 'Togo': 'TGO',
        'Barbados': 'BRB', 'Sierra Leone': 'SLE', 'Cayman Islands': 'CYM', 'Madagascar': 'MDG',
        'Jersey': 'JEY', 'Equatorial Guinea': 'GNQ', 'Turkmenistan': 'TKM', 'Liberia': 'LBR',
        'Gibraltar': 'GIB', 'Botswana': 'BWA', 'Guam': 'GUM', 'Gambia': 'GMB',
        'Ivory Coast': 'CIV', 'Uzbekistan': 'UZB', 'Mongolia': 'MNG', 'Martinique': 'MTQ',
        'Bhutan': 'BTN', 'Panama': 'PAN', 'Mauritania': 'MRT', 'Papua New Guinea': 'PNG',
        'Kiribati': 'KIR', 'Paraguay': 'PRY', 'Grenada': 'GRD', 'Guyana': 'GUY',
        'Nauru': 'NRU', 'Macao': 'MAC', 'Benin': 'BEN', 'Samoa': 'WSM',
        'Guadeloupe': 'GLP', 'Andorra': 'AND', 'Anguilla': 'AIA', 'Solomon Islands': 'SLB',
        'Marshall Islands': 'MHL', 'Pitcairn': 'PCN', 'Liechtenstein': 'LIE', 'Mayotte': 'MYT',
        'Christmas Island': 'CXR', 'Lao': 'LAO', 'Cocos Islands': 'CCK', 'Tuvalu': 'TUV',
        'Saint Helena': 'SHN', 'Saint Pierre and Miquelon': 'SPM', 'Falkland Islands': 'FLK',
        'Norfolk Island': 'NFK', 'Svalbard and Jan Mayen': 'SJM', 'Cook Islands': 'COK',
        'Niue': 'NIU', 'Tokelau': 'TKL', 'Holy See': 'VAT', 'British Indian Ocean Territory': 'IOT',
        'Western Sahara': 'ESH', 'Wallis and Futuna': 'WLF', 'French Southern Territories': 'ATF',
        'Heard Island and McDonald Islands': 'HMD', 'Saint Martin': 'MAF',
        'Saint Vincent and the Grenadines': 'VCT', 'Saint Kitts and Nevis': 'KNA',
        'Saint Lucia': 'LCA',
        'Sao Tome and Principe': 'STP'
    }

    
    sentiment_counts['iso_alpha'] = sentiment_counts['country'].map(country_to_iso)

    # Check for countries that were not mapped correctly
    unmapped_countries = sentiment_counts[sentiment_counts['iso_alpha'].isnull()]['country'].unique()
    
    # Print to help debug the mapping
    print("Mapped DataFrame:", sentiment_counts)

    # Print all countries and their ISO codes for verification
    for country in pycountry.countries:
        print(f"{country.name}: {country.alpha_3}")

    all_countries = pd.DataFrame({'country': list(country_to_iso.keys()), 'iso_alpha': list(country_to_iso.values())})
    all_countries = all_countries.merge(sentiment_counts[['country', 'primary_sentiment', 'positive_pct', 'negative_pct']], on='country', how='left').fillna({'primary_sentiment': 'neutral', 'positive_pct': 0, 'negative_pct': 0})

    fig = px.choropleth(all_countries, locations='iso_alpha', color='primary_sentiment',
                        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'},
                        title='World Sentiment Map',
                        hover_name='country',
                        hover_data={'positive_pct': ':.2f', 'negative_pct': ':.2f'})
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br><br>Positive: %{customdata[0]:.2f}%<br>Negative: %{customdata[1]:.2f}%')
    fig.update_geos(projection_type='natural earth')
    fig.update_layout(clickmode='event+select')

    # Update layout to adjust the size of the map
    fig.update_layout(
        width=1200,  # Width of the figure
        height=800   # Height of the figure
    )

    return pio.to_html(fig, full_html=False)

if __name__ == '__main__':
    app.run(debug=True)
