import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from rapidfuzz import fuzz
from word2number import w2n
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import Counter
import numpy as np
st.title('Generative Functions')
def loadTransactionDataDf():
    #transactionDataDf
    return pd.read_csv('C:/Users/gztan/Datasets/transaction_data.csv', encoding='utf-8')
def loadTransactionItemsDf():
    #transactionItemsDf
    return pd.read_csv ('C:/Users/gztan/Datasets/transaction_items.csv', encoding='utf-8')
def loadMerchantDf():
    #merchantDf
    return pd.read_csv ('C:/Users/gztan/Datasets/merchant.csv', encoding='utf-8')
def loadKeywordsDf():
    #keywordsDf
    return pd.read_csv ('C:/Users/gztan/Datasets/keywords.csv', encoding='utf-8')
def loadItemsDf():  
    #itemsDf  
    return pd.read_csv ('C:/Users/gztan/Datasets/items.csv', encoding='utf-8')
merchantID = st.text_input("Enter your merchant ID:")
if merchantID:
    st.write("Your merchant ID:", merchantID)
    location=loadMerchantDf().loc[loadMerchantDf()['merchant_id'] == merchantID, 'city_id'].values[0]
    st.write("Your location:", location)
    def convert_to_number(value):
        try:
            # Try to convert the value to an integer if it's a numeric string (e.g. "8")
            return int(value)
        except ValueError:
            try:
                # Try to convert the word to a number (e.g. "eight" -> 8)
                return w2n.word_to_num(value)
            except ValueError:
                return "Invalid input: could not convert to number."
    
    def popularMerchant(location,numberOfPopularMerchants):
        numberOfPopularMerchants=convert_to_number(numberOfPopularMerchants)
        location=str(location)
        if numberOfPopularMerchants<0:
            st.error(f"Please enter a valid number")
            return
        transactionDataDf = loadTransactionDataDf()
        merchantDf = loadMerchantDf()
        merchantDf['city_id'] = merchantDf['city_id'].astype(str)
        filteredMerchantsByCity = merchantDf[merchantDf['city_id'] == location]['merchant_id']
        filteredTransactions = transactionDataDf[transactionDataDf['merchant_id'].isin(filteredMerchantsByCity)]

        # Get the top 5 merchants by transaction frequency
        numberOfRecords=filteredTransactions['merchant_id'].value_counts()
        frequency = filteredTransactions['merchant_id'].value_counts().head(numberOfPopularMerchants)
        popularMerchants = []
        
        # Collect merchant name and transaction frequency
        for merchant_id, freq in frequency.items():
            merchant_name = merchantDf.loc[merchantDf['merchant_id'] == merchant_id, 'merchant_name'].values[0]
            popularMerchants.append([merchant_name, freq])
        if len(numberOfRecords) < numberOfPopularMerchants:
            if len(numberOfRecords)==1:
                st.write(f"There is only {len(numberOfRecords)} merchants at your location.")
            else:
                st.write(f"There are only {len(numberOfRecords)} merchants at your location.")
        else:
            st.write("The popular merchants that has most frequent sales at your location are:")

        # Display the merchants with indentation
        i = 1
        for merchant, freq in popularMerchants:
            st.text(f"{i}.    {merchant}    ({freq} transactions)")
            i += 1

    def mergeKeyword():
        keywordsDf = loadKeywordsDf()
        itemsDf = loadItemsDf()

        # Clean strings
        itemsDf['item_name_clean'] = itemsDf['item_name'].str.lower().str.strip()
        keywordsDf['keyword_clean'] = keywordsDf['keyword'].str.lower().str.strip()

        # Combine all text for TF-IDF
        all_text = itemsDf['item_name_clean'].tolist() + keywordsDf['keyword_clean'].tolist()

        # Generate TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_text)

        # Split back
        item_vectors = tfidf_matrix[:len(itemsDf)]
        keyword_vectors = tfidf_matrix[len(itemsDf):]

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(item_vectors, keyword_vectors)

        # Match based on threshold
        threshold = 0.6  # You can tune this
        matched_keywords = []
        for i, row in enumerate(similarity_matrix):
            matched = [keywordsDf.iloc[j]['keyword'] for j, score in enumerate(row) if score >= threshold]
            matched_keywords.append(matched)

        itemsDf['matched_keywords'] = matched_keywords
        exploded = itemsDf.explode('matched_keywords').dropna(subset=['matched_keywords'])

        # Merge stats
        merged = exploded.merge(
            keywordsDf[['keyword', 'view', 'menu', 'checkout', 'order']],
            left_on='matched_keywords',
            right_on='keyword',
            how='left'
        )

        # Aggregate and return
        stats = merged.groupby('item_name_clean')[['view', 'menu', 'checkout', 'order']].sum().reset_index()
        result = itemsDf.merge(stats, on='item_name_clean', how='left')

        for col in ['view', 'menu', 'checkout', 'order']:
            result[col] = result[col].fillna(0).astype(int)

        return result.drop(columns=['item_name_clean'])
    def getItemsKeywordsAtLocation(location):
        itemsKeywordViewDf = mergeKeyword()
        merchantDf=loadMerchantDf()
        merchant_ids_in_city = merchantDf[merchantDf['city_id'] == location]['merchant_id']
        itemsKeywordViewDf = itemsKeywordViewDf[
            itemsKeywordViewDf['merchant_id'].isin(merchant_ids_in_city)
        ]
        return itemsKeywordViewDf
    def viewInterpretion(merchantID):
        # Merge keyword data (assuming mergeKeyword is a function you have defined)
        itemsKeywordDf = mergeKeyword()
        # Filter the dataframe based on the given merchant ID
        filteredDf = itemsKeywordDf[itemsKeywordDf['merchant_id'] == merchantID]
        
        # Calculate the average views for the merchant
        averageViews = filteredDf['view'].mean()

        # Step 3: Apply KMeans clustering on the entire 'view' feature of the whole dataset
        kmeans = KMeans(n_clusters=3, random_state=42,n_init='auto')
        itemsKeywordDf['view_category'] = kmeans.fit_predict(itemsKeywordDf[['view']])

        # Step 4: Find the cluster of the merchant's average views based on the overall 'view' distribution
        # Predict the cluster for the merchant's average view
        average_value_cluster = kmeans.predict([[averageViews]])

        # Step 5: Interpret the results based on the cluster
        doingPerformance = ""
        
        if average_value_cluster[0] == 0:
            doingPerformance += "it looks like the exposure rate is lower than expected. You may subscribe to Grab Plus to boost your reputation. Increasing exposure will help increase brand awareness and customer acquisition, ultimately leading to higher conversions and long-term success."
        elif average_value_cluster[0] == 1:
            doingPerformance += "the exposure rate is currently at a moderate level, which is a solid foundation. You're reaching a decent portion of your target audience, but there is still room for growth to maximize visibility. With some additional effort in areas like targeted advertising, partnerships, or expanding content, we could push that exposure rate higher to ensure you're reaching even more potential customers."
        else:
            doingPerformance += "your exposure rate is really strong right now, which is fantastic! This means that your brand/product is reaching a wide audience, increasing visibility and the likelihood of conversions. To keep this momentum going, it’s important to maintain and possibly even enhance this exposure with regular engagement and targeted strategies that keep the audience interested."

        # Step 6: Display the interpretation and the filtered dataframe in Streamlit
        st.write(f'Your store has an average view of {averageViews:.2f} and {doingPerformance}')
        # Get all merchants in the selected location
        totalDf=getItemsKeywordsAtLocation(location)
        view_data = totalDf['view'].dropna().values.reshape(-1, 1)

        # Optional: only cluster if enough data points
        if len(view_data) >= 3:
            # Step 2: Fit KMeans on total view data
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            kmeans.fit(view_data)

            # Step 3: Predict clusters for filteredDf
            filteredDf = filteredDf.copy()  # avoid SettingWithCopyWarning
            filteredDf['view_cluster'] = kmeans.predict(filteredDf['view'].values.reshape(-1, 1))

            # Step 4: Map cluster labels to names based on center order (e.g., low, medium, high)
            centers = kmeans.cluster_centers_.flatten()
            ordered_labels = np.argsort(centers)
            label_map = {ordered_labels[0]: 'low_view', ordered_labels[1]: 'medium_view', ordered_labels[2]: 'high_view'}

            filteredDf['view_cluster_label'] = filteredDf['view_cluster'].map(label_map)

        else:
            filteredDf['view_cluster_label'] = 'not_enough_data'
        filteredDf = filteredDf.copy()
        # Calculate order/view ratio
        filteredDf['order_view_ratio'] = filteredDf['order'] / filteredDf['view']

        # Filter rows where order <= 10% of view
        lowConversionDf = filteredDf[filteredDf['order_view_ratio'] <= 0.01]
        if not lowConversionDf.empty:
            message = "⚠️ The following items have low order-to-view conversion (≤ 1%):\n\n"
            for _, row in lowConversionDf.iterrows():
                message += f"- Merchant {row['merchant_id']} has {int(row['order'])} orders and {int(row['view'])} views (conversion: {round(row['order_view_ratio']*100, 2)}%)\n"
            st.text(message)
        else:
            st.text("✅ All items have a healthy order-to-view conversion rate.")
        st.dataframe(filteredDf[['item_id','item_name','item_price','view','menu','checkout','order','view_cluster_label']])
        
    def viewMostViewedFood(merchantID, numResultsFoodView):
        # Merge keyword data (assuming mergeKeyword is a function you have defined)
        itemsKeywordDf = mergeKeyword()
        
        # Convert numResultsFoodView to a number (if necessary)
        numResultsFoodView = convert_to_number(numResultsFoodView)
        
        # Filter the dataframe based on the given merchant ID
        filteredDf = itemsKeywordDf[itemsKeywordDf['merchant_id'] == merchantID]
        
        # Sort the DataFrame by 'view' in descending order
        sorted_filteredDf = filteredDf.sort_values(by='view', ascending=False)
        if len(sorted_filteredDf) < numResultsFoodView:
            if len(sorted_filteredDf)==1:
                st.write(f"There is only {len(sorted_filteredDf)} food in your shop.")
            else:
                st.write(f"There are only {len(sorted_filteredDf)} food in your shop.")
        else:
            st.write("The most viewed food in your shop are: ")
        # Iterate through the sorted DataFrame and print results
        for i, (idx, row) in enumerate(sorted_filteredDf.iterrows(), start=1):
            item_name = row['item_name']
            view = row['view']
            st.write(f"{i}. {item_name} - ({view} views)")
            
            # Stop if the required number of results is reached
            if i == numResultsFoodView:
                break
    def analyzeFoodType(merchantID, location):
        # Load data
        itemsKeywordAnalyzeFoodTypeDf = getItemsKeywordsAtLocation(location)

        # Filter current merchant data
        filteredDf = itemsKeywordAnalyzeFoodTypeDf[itemsKeywordAnalyzeFoodTypeDf['merchant_id'] == merchantID]

        # Calculate cuisine tag percentages for the merchant
        value_percentages = (filteredDf['cuisine_tag'].value_counts(normalize=True) * 100).round(2)
        mode_array = np.array([[tag, pct] for tag, pct in value_percentages.items()])

        st.write('Your shop is selling these type of food:')
        i = 1
        storeTypeAndSell = []

        # Prepare the output DataFrame
        filteredDf = filteredDf.copy()
        filteredDf['price_cluster_label'] = None

        # Count total unique merchants in city for competition calculation
        totalNumMerchants = itemsKeywordAnalyzeFoodTypeDf['merchant_id'].nunique()
        strOutput = ""

        for foodType, percentage in mode_array:
            # Filter city-wide items of the same cuisine
            filtered_cuisine = itemsKeywordAnalyzeFoodTypeDf[itemsKeywordAnalyzeFoodTypeDf['cuisine_tag'] == (foodType)]

            # Count number of unique merchants selling this cuisine
            merchant_frequency = filtered_cuisine['merchant_id'].nunique()
            storeTypeAndSell.append([foodType, merchant_frequency])

            st.text(f"{i}.    {foodType}    ({percentage} %)")
            i += 1

            # Competition level
            competition = merchant_frequency / totalNumMerchants * 100
            if competition < 20:
                strOutput += f"The competition of food type {foodType} is low. There are only {merchant_frequency} merchants selling {foodType} at your location. "
            elif competition < 60:
                strOutput += f"The competition of food type {foodType} is moderate. There are {merchant_frequency} merchants selling {foodType} at your location. "
            else:
                strOutput += f"The competition of food type {foodType} is very high. There are a huge number of {merchant_frequency} merchants selling {foodType} at your location. "

            # --- Price Clustering ---
            city_prices = filtered_cuisine['item_price'].values.reshape(-1, 1)

            if len(city_prices) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                kmeans.fit(city_prices)

                # Get cluster label mapping
                cluster_order = sorted(range(3), key=lambda i: kmeans.cluster_centers_[i][0])
                label_map = {cluster_order[0]: 'cheap', cluster_order[1]: 'moderate', cluster_order[2]: 'expensive'}

                # Predict clusters for this merchant's items with the same cuisine
                mask = filteredDf['cuisine_tag'] == (foodType)
                shop_prices = filteredDf.loc[mask, 'item_price'].values.reshape(-1, 1)
                clusters = kmeans.predict(shop_prices)
                filteredDf.loc[mask, 'price_cluster_label'] = [label_map[c] for c in clusters]
            else:
                # If not enough city data for clustering
                filteredDf.loc[filteredDf['cuisine_tag'] == int(foodType), 'price_cluster_label'] = 'not_enough_data'

        st.write(strOutput)
        st.dataframe(filteredDf[['item_id','cuisine_tag','item_name','item_price','price_cluster_label']])
        
        
    #1) Find the popular merchants based on frequency of sales in current location
    st.subheader('1) Find the popular merchants that has the most frequent of sales at current location')
    numResults = st.text_input("Enter number of results:", key="numResult")
    if(numResults):
        popularMerchant(location,numResults)
    else:
        popularMerchant(location,5)
        
    #2) Interpret view (Number of search by users)
    st.subheader('2) Interpret view (Number of search by users)')
    viewInterpretion(merchantID)
        
    #3) Determine the most viewed food 
    st.subheader('3) Determine the most viewed food in your shop')
    numResultsFoodView = st.text_input("Enter number of results:", key="numResultsFoodView")
    if(numResultsFoodView):
        viewMostViewedFood(merchantID, numResultsFoodView)
    else:
        viewMostViewedFood(merchantID,5)
        
    #4) Analyze the food type and its competition at your location
    st.subheader('4) Analyze the food type and its competition at your location')
    analyzeFoodType(merchantID, location)
    
    