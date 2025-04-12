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
from datetime import datetime, timedelta
import google.generativeai as genai
import re

st.title("Generative Functions")


def loadTransactionDataDf():
    # transactionDataDf
    transactionDataDf = pd.read_csv("Datasets/transaction_data.csv", encoding="utf-8")
    transactionDataDf["order_time"] = pd.to_datetime(transactionDataDf["order_time"])
    transactionDataDf["order_day"] = transactionDataDf["order_time"].dt.strftime(
        "%d/%m/%Y"
    )
    return transactionDataDf


def loadTransactionItemsDf():
    # transactionItemsDf
    return pd.read_csv("Datasets/transaction_items.csv", encoding="utf-8")


def loadMerchantDf():
    # merchantDf
    return pd.read_csv("Datasets/merchant.csv", encoding="utf-8")


def loadKeywordsDf():
    # keywordsDf
    return pd.read_csv("Datasets/keywords.csv", encoding="utf-8")


def loadItemsDf():
    # itemsDf
    return pd.read_csv("Datasets/items.csv", encoding="utf-8")


merchantID = st.text_input("Enter your merchant ID:")
raw_date = st.text_input("Enter a date (day/month/year):")
todayDate = raw_date

if raw_date:
    try:
        # Try to parse the date regardless of single/double digit format
        parsed_date = datetime.strptime(raw_date, "%d/%m/%Y")
        # Format the date into "dd/mm/yyyy" with leading zeros
        todayDate = parsed_date.strftime("%d/%m/%Y")
    except ValueError:
        st.error("Invalid date format. Please enter the date as day/month/year.")

if merchantID:
    if todayDate:
        st.write("Your merchant ID:", merchantID)
        location = (
            loadMerchantDf()
            .loc[loadMerchantDf()["merchant_id"] == merchantID, "city_id"]
            .values[0]
        )

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

        def sliceTransactionByMerchant(merchantID):
            transactionDf = loadTransactionDataDf()
            return transactionDf[transactionDf["merchant_id"] == merchantID]

        def sliceTransactionByMerchantAndDate(merchantID, date):
            transactionDateDf = sliceTransactionByMerchant(merchantID)
            return transactionDateDf[transactionDateDf["order_day"] == date]

        def popularMerchant(location, numberOfPopularMerchants):
            numberOfPopularMerchants = convert_to_number(numberOfPopularMerchants)
            location = str(location)
            if numberOfPopularMerchants < 0:
                st.error(f"Please enter a valid number")
                return
            transactionDataDf = loadTransactionDataDf()
            merchantDf = loadMerchantDf()
            merchantDf["city_id"] = merchantDf["city_id"].astype(str)
            filteredMerchantsByCity = merchantDf[merchantDf["city_id"] == location][
                "merchant_id"
            ]
            filteredTransactions = transactionDataDf[
                transactionDataDf["merchant_id"].isin(filteredMerchantsByCity)
            ]

            # Get the top 5 merchants by transaction frequency
            numberOfRecords = filteredTransactions["merchant_id"].value_counts()
            frequency = (
                filteredTransactions["merchant_id"]
                .value_counts()
                .head(numberOfPopularMerchants)
            )
            popularMerchants = []

            # Collect merchant name and transaction frequency
            for merchant_id, freq in frequency.items():
                merchant_name = merchantDf.loc[
                    merchantDf["merchant_id"] == merchant_id, "merchant_name"
                ].values[0]
                popularMerchants.append([merchant_name, freq])
            if len(numberOfRecords) < numberOfPopularMerchants:
                if len(numberOfRecords) == 1:
                    st.write(
                        f"There is only {len(numberOfRecords)} merchants at your location."
                    )
                else:
                    st.write(
                        f"There are only {len(numberOfRecords)} merchants at your location."
                    )
            else:
                st.write(
                    "The popular merchants that has most frequent sales at your location are:"
                )

            # Display the merchants with indentation
            i = 1
            for merchant, freq in popularMerchants:
                st.text(f"{i}.    {merchant}    ({freq} transactions)")
                i += 1

        def mergeKeyword():
            keywordsDf = loadKeywordsDf()
            itemsDf = loadItemsDf()

            # Clean strings
            itemsDf["item_name_clean"] = itemsDf["item_name"].str.lower().str.strip()
            keywordsDf["keyword_clean"] = keywordsDf["keyword"].str.lower().str.strip()

            # Combine all text for TF-IDF
            all_text = (
                itemsDf["item_name_clean"].tolist()
                + keywordsDf["keyword_clean"].tolist()
            )

            # Generate TF-IDF matrix
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_text)

            # Split back
            item_vectors = tfidf_matrix[: len(itemsDf)]
            keyword_vectors = tfidf_matrix[len(itemsDf) :]

            # Compute cosine similarity
            similarity_matrix = cosine_similarity(item_vectors, keyword_vectors)

            # Match based on threshold
            threshold = 0.6  # You can tune this
            matched_keywords = []
            for i, row in enumerate(similarity_matrix):
                matched = [
                    keywordsDf.iloc[j]["keyword"]
                    for j, score in enumerate(row)
                    if score >= threshold
                ]
                matched_keywords.append(matched)

            itemsDf["matched_keywords"] = matched_keywords
            exploded = itemsDf.explode("matched_keywords").dropna(
                subset=["matched_keywords"]
            )

            # Merge stats
            merged = exploded.merge(
                keywordsDf[["keyword", "view", "menu", "checkout", "order"]],
                left_on="matched_keywords",
                right_on="keyword",
                how="left",
            )

            # Aggregate and return
            stats = (
                merged.groupby("item_name_clean")[["view", "menu", "checkout", "order"]]
                .sum()
                .reset_index()
            )
            result = itemsDf.merge(stats, on="item_name_clean", how="left")

            for col in ["view", "menu", "checkout", "order"]:
                result[col] = result[col].fillna(0).astype(int)

            return result.drop(columns=["item_name_clean"])

        def getItemsKeywordsAtLocation(location):
            itemsKeywordViewDf = mergeKeyword()
            merchantDf = loadMerchantDf()
            merchant_ids_in_city = merchantDf[merchantDf["city_id"] == location][
                "merchant_id"
            ]
            itemsKeywordViewDf = itemsKeywordViewDf[
                itemsKeywordViewDf["merchant_id"].isin(merchant_ids_in_city)
            ]
            return itemsKeywordViewDf

        def viewInterpretation(merchantID):
            import numpy as np
            import pandas as pd
            import streamlit as st
            from sklearn.cluster import KMeans

            # Merge keyword data (assuming mergeKeyword is defined)
            itemsKeywordDf = mergeKeyword()
            # Filter the dataframe based on the given merchant ID
            filteredDf = analyzeFoodType(merchantID, location)

            # Calculate the average views for the merchant
            averageViews = filteredDf["view"].mean()

            # Step 3: Apply KMeans clustering on the entire 'view' feature of the whole dataset
            # (Here we use n_init='auto' per your updated code, if your sklearn version supports it)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
            itemsKeywordDf["view_category"] = kmeans.fit_predict(
                itemsKeywordDf[["view"]]
            )

            # Step 4: Find the cluster of the merchant's average views based on the overall 'view' distribution
            average_value_cluster = kmeans.predict([[averageViews]])

            # Step 5: Interpret the results based on the cluster
            doingPerformance = ""
            if average_value_cluster[0] == 0:
                doingPerformance += (
                    "It looks like the exposure rate is lower than expected. "
                    "You may subscribe to Grab Plus to boost your reputation. "
                    "Increasing exposure will help increase brand awareness and customer acquisition, "
                    "ultimately leading to higher conversions and long-term success."
                )
            elif average_value_cluster[0] == 1:
                doingPerformance += (
                    "The exposure rate is currently at a moderate level, which is a solid foundation. "
                    "You're reaching a decent portion of your target audience, but there is still room for growth "
                    "to maximize visibility. With some additional effort in areas like targeted advertising, partnerships, "
                    "or expanding content, you could push that exposure rate higher to reach even more potential customers."
                )
            else:
                doingPerformance += (
                    "Your exposure rate is really strong right now, which is fantastic! This means that your brand "
                    "or product is reaching a wide audience, increasing visibility and the likelihood of conversions. "
                    "To keep this momentum going, it’s important to maintain and possibly even enhance this exposure "
                    "with regular engagement and targeted strategies that keep the audience interested."
                )

            st.write(
                f"Your store has an average view of {averageViews:.2f} and {doingPerformance}"
            )

            # Get all merchants in the selected location
            totalDf = getItemsKeywordsAtLocation(location)
            view_data = totalDf["view"].dropna().values.reshape(-1, 1)

            # Optional: only cluster if enough data points
            if len(view_data) >= 3:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                kmeans.fit(view_data)

                filteredDf = filteredDf.copy()  # avoid SettingWithCopyWarning
                filteredDf["view_cluster"] = kmeans.predict(
                    filteredDf["view"].values.reshape(-1, 1)
                )

                # Map cluster labels to names based on center order (e.g., low, medium, high)
                centers = kmeans.cluster_centers_.flatten()
                ordered_labels = np.argsort(centers)
                label_map = {
                    ordered_labels[0]: "low_view",
                    ordered_labels[1]: "medium_view",
                    ordered_labels[2]: "high_view",
                }

                filteredDf["view_cluster_label"] = filteredDf["view_cluster"].map(
                    label_map
                )
            else:
                filteredDf["view_cluster_label"] = "not_enough_data"

            # Calculate order/view ratio
            filteredDf["order_view_ratio"] = filteredDf["order"] / filteredDf["view"]

            # Create new column for order-to-view conversion category
            # Example thresholds: < 1% = low, 1%-3% = medium, >=3% = high.
            def conversion_category(ratio):
                if ratio < 0.01:
                    return "low"
                elif ratio < 0.03:
                    return "medium"
                else:
                    return "high"

            filteredDf["order_to_view_conversion"] = filteredDf[
                "order_view_ratio"
            ].apply(conversion_category)

            # Filter rows where order is <= 1% of view and create a text message
            lowConversionDf = filteredDf[filteredDf["order_view_ratio"] <= 0.01]
            if not lowConversionDf.empty:
                message = "⚠️ The following items have low order-to-view conversion (≤ 1%):\n\n"
                for _, row in lowConversionDf.iterrows():
                    message += (
                        f"- {row['item_name']} has {int(row['order'])} orders and {int(row['view'])} views "
                        f"(conversion: {round(row['order_view_ratio']*100, 2)}%)\n"
                    )
                    if row["price_cluster_label"] == "expensive":
                        message += f"The price of {row['item_name']} is too high compared with other merchants.\n\n"
                    else:
                        message += "\n"
                st.text(message)
            else:
                st.text("✅ All items have a healthy order-to-view conversion rate.")

            # Display a subset of columns in filteredDf including the new conversion category.
            st.dataframe(
                filteredDf[
                    [
                        "item_id",
                        "item_name",
                        "item_price",
                        "view",
                        "menu",
                        "checkout",
                        "order",
                        "view_cluster_label",
                        "order_to_view_conversion",
                    ]
                ]
            )

        def viewMostViewedFood(merchantID, numResultsFoodView):
            # Merge keyword data (assuming mergeKeyword is a function you have defined)
            itemsKeywordDf = mergeKeyword()

            # Convert numResultsFoodView to a number (if necessary)
            numResultsFoodView = convert_to_number(numResultsFoodView)

            # Filter the dataframe based on the given merchant ID
            filteredDf = itemsKeywordDf[itemsKeywordDf["merchant_id"] == merchantID]

            # Sort the DataFrame by 'view' in descending order
            sorted_filteredDf = filteredDf.sort_values(by="view", ascending=False)
            if len(sorted_filteredDf) < numResultsFoodView:
                if len(sorted_filteredDf) == 1:
                    st.write(
                        f"There is only {len(sorted_filteredDf)} food in your shop."
                    )
                else:
                    st.write(
                        f"There are only {len(sorted_filteredDf)} food in your shop."
                    )
            else:
                st.write("The most viewed food in your shop are: ")
            # Iterate through the sorted DataFrame and print results
            for i, (idx, row) in enumerate(sorted_filteredDf.iterrows(), start=1):
                item_name = row["item_name"]
                view = row["view"]
                st.write(f"{i}. {item_name} - ({view} views)")

                # Stop if the required number of results is reached
                if i == numResultsFoodView:
                    break

        def analyzeFoodType(merchantID, location):
            # Load data
            itemsKeywordAnalyzeFoodTypeDf = getItemsKeywordsAtLocation(location)

            # Filter current merchant data
            filteredDf = itemsKeywordAnalyzeFoodTypeDf[
                itemsKeywordAnalyzeFoodTypeDf["merchant_id"] == merchantID
            ]

            # Calculate cuisine tag percentages for the merchant
            value_percentages = (
                filteredDf["cuisine_tag"].value_counts(normalize=True) * 100
            ).round(2)
            mode_array = np.array(
                [[tag, pct] for tag, pct in value_percentages.items()]
            )

            st.write("Your shop is selling these type of food:")
            i = 1
            storeTypeAndSell = []

            # Prepare the output DataFrame
            filteredDf = filteredDf.copy()
            filteredDf["price_cluster_label"] = None

            # Count total unique merchants in city for competition calculation
            totalNumMerchants = itemsKeywordAnalyzeFoodTypeDf["merchant_id"].nunique()
            strOutput = ""

            for foodType, percentage in mode_array:
                # Filter city-wide items of the same cuisine
                filtered_cuisine = itemsKeywordAnalyzeFoodTypeDf[
                    itemsKeywordAnalyzeFoodTypeDf["cuisine_tag"] == (foodType)
                ]

                # Count number of unique merchants selling this cuisine
                merchant_frequency = filtered_cuisine["merchant_id"].nunique()
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
                city_prices = filtered_cuisine["item_price"].values.reshape(-1, 1)

                if len(city_prices) >= 3:
                    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                    kmeans.fit(city_prices)

                    # Get cluster label mapping
                    cluster_order = sorted(
                        range(3), key=lambda i: kmeans.cluster_centers_[i][0]
                    )
                    label_map = {
                        cluster_order[0]: "cheap",
                        cluster_order[1]: "moderate",
                        cluster_order[2]: "expensive",
                    }

                    # Predict clusters for this merchant's items with the same cuisine
                    mask = filteredDf["cuisine_tag"] == (foodType)
                    shop_prices = filteredDf.loc[mask, "item_price"].values.reshape(
                        -1, 1
                    )
                    clusters = kmeans.predict(shop_prices)
                    filteredDf.loc[mask, "price_cluster_label"] = [
                        label_map[c] for c in clusters
                    ]
                else:
                    # If not enough city data for clustering
                    filteredDf.loc[
                        filteredDf["cuisine_tag"] == (foodType), "price_cluster_label"
                    ] = "not_enough_data"

            st.write(strOutput)
            st.dataframe(
                filteredDf[
                    [
                        "item_id",
                        "cuisine_tag",
                        "item_name",
                        "item_price",
                        "price_cluster_label",
                    ]
                ]
            )
            return filteredDf

        def analyzeDailySales(merchantID, todayDate, numResults):
            merchantDateDf = sliceTransactionByMerchantAndDate(merchantID, todayDate)
            if merchantDateDf.empty:
                st.warning(
                    f"No sales data found for Merchant {merchantID} on {todayDate}."
                )
                return  # Exit early if no data
            transactionItemsDf = loadTransactionItemsDf()
            itemsDf = loadItemsDf()
            merchantDateDf["order_time"] = pd.to_datetime(merchantDateDf["order_time"])
            # Round down the time to the nearest 2-hour block
            merchantDateDf["sales_time"] = (
                merchantDateDf["order_time"].dt.floor("1H").dt.strftime("%H%M")
            )
            import plotly.express as px

            # Aggregate
            sales_by_time = merchantDateDf.groupby("sales_time", as_index=False)[
                "order_value"
            ].sum()
            sales_by_time = sales_by_time.sort_values("sales_time")

            # Plot
            fig = px.line(
                sales_by_time,
                x="sales_time",
                y="order_value",
                labels={
                    "sales_time": "Sales Time (1-hour slots)",
                    "order_value": "Total Order Value",
                },
                title="Sales Distribution by Time Slot",
                markers=True,
            )

            import uuid

            unique_key = f"plot_{merchantID}_{todayDate}_{uuid.uuid4()}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            merchantDateDf["total_sales"] = merchantDateDf.groupby("sales_time")[
                "order_value"
            ].transform("sum")
            top_sales_times = (
                merchantDateDf.groupby("sales_time")["order_value"]
                .sum()
                .sort_values(ascending=False)
                .head(numResults)
            )
            st.write(f"Top {numResults} Sales Time Slots with Highest Total Sales:")
            for sales_time, total in top_sales_times.items():
                # Get end time (add 2 hours)
                hour = int(sales_time[:2])
                end_hour = (hour + 2) % 24
                time_range = f"{sales_time}-{end_hour:02}00"
                st.write(f"⏰ {time_range}: RM {total:.2f}")
            st.write(f"Your daily sales on ({todayDate}):")
            merchantDateDf = merchantDateDf.sort_values(by="order_time", ascending=True)
            # Step 1: Merge merchant data with transaction items to get item_id per order
            merged_df = pd.merge(
                merchantDateDf,
                transactionItemsDf[["order_id", "item_id"]],
                on="order_id",
                how="left",
            )

            # Step 2: Merge with itemsDf to get item_name
            merged_df = pd.merge(
                merged_df, itemsDf[["item_id", "item_name"]], on="item_id", how="left"
            )

            # Step 3 (Optional): Drop columns you no longer need (like order_id or item_id)
            merged_df = merged_df.drop(columns=["item_id"])

            # Step 4 (Optional): Reorder columns if you want item_name first
            cols = ["item_name"] + [
                col for col in merged_df.columns if col != "item_name"
            ]
            merged_df = merged_df[cols]

            st.dataframe(
                merged_df[
                    ["order_id", "item_name", "order_time", "order_value", "eater_id"]
                ]
            )
            salesAtTimeSlotView = st.text_input("Enter time to check sales (e.g., 14:00):",key="salesAtTimeSlotView")
            if salesAtTimeSlotView:
                printSalesAtTimeSlotDf(merchantID, todayDate, salesAtTimeSlotView)

        def printSalesAtTimeSlotDf(merchantID, todayDate, salesAtTimeSlotView):
            merchantDateDf = sliceTransactionByMerchantAndDate(merchantID, todayDate)
            if merchantDateDf.empty:
                st.warning(
                    f"No sales data found for Merchant {merchantID} on {todayDate}."
                )
                return  # Exit early if no data
            transactionItemsDf = loadTransactionItemsDf()
            itemsDf = loadItemsDf()
            merchantDateDf["order_time"] = pd.to_datetime(merchantDateDf["order_time"])
            # Round down the time to the nearest 2-hour block
            merchantDateDf["sales_time"] = (
                merchantDateDf["order_time"].dt.floor("1H").dt.strftime("%H%M")
            )

            # Aggregate
            sales_by_time = merchantDateDf.groupby("sales_time", as_index=False)[
                "order_value"
            ].sum()
            sales_by_time = sales_by_time.sort_values("sales_time")
            # Get filtered data (assuming analyzeDailySales returns a DataFrame)

            filteredDf = merchantDateDf

            def to_hour_block(time_str):
                """
                Accepts '0855', '8:55', '08:55', etc.
                Returns a string like '0800' (floored to the hour).
                """
                # Normalize input: insert colon if missing
                if ":" not in time_str:
                    time_str = time_str.zfill(4)  # Ensure 4 digits
                    time_str = time_str[:2] + ":" + time_str[2:]

                time_obj = datetime.strptime(time_str, "%H:%M")
                return f"{time_obj.hour:02}00"

            # Format as '0800'
            formatted_time = to_hour_block(salesAtTimeSlotView)

            # Filter by sales_time
            filteredDf = filteredDf[filteredDf["sales_time"] == formatted_time]

            # Display result
            st.write(f"🕒 Sales for time slot starting at {formatted_time}:")
            filteredDf = filteredDf.sort_values(by="order_time", ascending=True)
            # Step 1: Merge merchant data with transaction items to get item_id per order
            merged_df = pd.merge(
                filteredDf,
                transactionItemsDf[["order_id", "item_id"]],
                on="order_id",
                how="left",
            )

            # Step 2: Merge with itemsDf to get item_name
            merged_df = pd.merge(
                merged_df, itemsDf[["item_id", "item_name"]], on="item_id", how="left"
            )

            # Step 3 (Optional): Drop columns you no longer need (like order_id or item_id)
            merged_df = merged_df.drop(columns=["item_id"])

            # Step 4 (Optional): Reorder columns if you want item_name first
            cols = ["item_name"] + [
                col for col in merged_df.columns if col != "item_name"
            ]
            merged_df = merged_df[cols]

            st.dataframe(
                merged_df[
                    ["order_id", "item_name", "order_time", "order_value", "eater_id"]
                ]
            )

        def analyzeMostOrderedFood(merchantID, todayDate, numMostOrderedFood=5):
            numMostOrderedFood = int(numMostOrderedFood)

            # Step 1: Load and filter merchant transactions
            merchantDateDf = sliceTransactionByMerchant(merchantID)
            if merchantDateDf.empty:
                st.warning("No transaction data found for this merchant.")
                return

            merchantDateDf["order_day"] = pd.to_datetime(
                merchantDateDf["order_day"], format="%d/%m/%Y"
            )
            today = pd.to_datetime(todayDate, format="%d/%m/%Y")

            merchantDateDf = merchantDateDf[
                (merchantDateDf["order_day"].dt.month == today.month)
                & (merchantDateDf["order_day"].dt.year == today.year)
            ]
            if merchantDateDf.empty:
                st.warning("No transaction data for the given month and year.")
                return

            # Step 2: Create sales_time in "HHMM-HHMM" format
            merchantDateDf["order_time"] = pd.to_datetime(merchantDateDf["order_time"])
            rounded_time = merchantDateDf["order_time"].dt.floor("1H")
            merchantDateDf["sales_time"] = (
                rounded_time.dt.strftime("%H%M")
                + "-"
                + (rounded_time + pd.Timedelta(hours=1)).dt.strftime("%H%M")
            )
            merchantDateDf["order_day"] = merchantDateDf["order_time"].dt.strftime(
                "%d/%m/%Y"
            )

            # Step 3: Load transaction items
            transactionItemsDf = loadTransactionItemsDf()
            if transactionItemsDf.empty:
                st.warning("No transaction items data available.")
                return

            # Step 4: Merge order items with sales time
            merged_df = pd.merge(
                transactionItemsDf,
                merchantDateDf[["order_id", "sales_time"]],
                on="order_id",
                how="inner",
            )

            # Step 5: Frequency calculation
            freq_df = (
                merged_df.groupby(["sales_time", "item_id"])
                .size()
                .reset_index(name="frequency")
            )

            # Step 6: Clustering on frequency
            if len(freq_df) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                freq_df["cluster"] = kmeans.fit_predict(freq_df[["frequency"]])
                most_freq_cluster = (
                    freq_df.groupby("cluster")["frequency"].mean().idxmax()
                )
                filtered_freq_df = freq_df[freq_df["cluster"] == most_freq_cluster]
            else:
                filtered_freq_df = freq_df  # Not enough data to cluster

            # Step 7: Add item_name and display format
            itemsDf = (
                loadItemsDf()
            )  # Make sure this function returns your items metadata with item_id + item_name
            id_to_name = dict(zip(itemsDf["item_id"], itemsDf["item_name"]))
            filtered_freq_df["item_name"] = filtered_freq_df["item_id"].map(id_to_name)
            filtered_freq_df["display_name"] = (
                filtered_freq_df["item_name"]
                + " ("
                + filtered_freq_df["frequency"].astype(str)
                + ")"
            )

            # Step 8: Sort and select top N items per time block
            sorted_items = filtered_freq_df.sort_values(
                by=["sales_time", "frequency"], ascending=[True, False]
            )
            top_items = (
                sorted_items.groupby("sales_time")
                .head(numMostOrderedFood)
                .groupby("sales_time")["display_name"]
                .apply(list)
            )

            # Step 9: Pad lists to same length
            max_len = top_items.apply(len).max()
            padded_lists = {
                time: lst + [None] * (max_len - len(lst))
                for time, lst in top_items.items()
            }
            final_df_named = pd.DataFrame.from_dict(
                padded_lists, orient="index"
            ).transpose()

            # Step 10: Add ordinal index
            def ordinal(n):
                return "%d%s" % (
                    n,
                    "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
                )

            final_df_named.index = [ordinal(i + 1) for i in range(len(final_df_named))]

            # Step 11: Show the result
            st.dataframe(final_df_named)

        # Configure Gemini API
        def setup_gemini():
            if "GEMINI_API_KEY" in st.secrets:
                gemini_api_key = st.secrets["GEMINI_API_KEY"]
            else:
                gemini_api_key = st.session_state.get("gemini_api_key", "")
                if not gemini_api_key:
                    gemini_api_key = st.text_input(
                        "Enter your Gemini API Key:", type="password"
                    )
                    if gemini_api_key:
                        st.session_state.gemini_api_key = gemini_api_key

            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                return True
            return False

        # Get response from Gemini
        def get_gemini_response(prompt):
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                system_context = """
                You are GAI 4.0o, a helpful multilingual assistant from Grab company for a food merchant analytics app.
                
                You can help users with:
                1. Finding top viewed food items
                2. Finding popular merchants
                3. Analyzing exposure rates
                4. Analyzing food types
                5. Analyzing daily sales
                6. Analyzing most ordered food items
                7. Reply to user by using some general sales advice when they ask for boost or improve their performance
                
                Keep responses friendly by using some emoji, concise and helpful. Suggest specific actions the user can take.
                Also, you should be able to prompt user to enter correct input when they enter in other language except English (also, you need to reply them using the language used in their input), such as "最畅销的食物", you should recommend the user to enter "most ordered food"
                """
                full_prompt = (
                    f"{system_context}\n\nUser query: {prompt}\n\nYour response:"
                )
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                return f"Error: {str(e)}"

        # Extract action and number
        def extract_action_and_number(input_text):
            input_text = input_text.lower()
            number = None
            number_match = re.search(r"\b\d+\b", input_text)
            if number_match:
                number = int(number_match.group())

            if any(
                k in input_text
                for k in ["top food", "most viewed food", "best food", "popular food", "food views"]
            ):
                return "most_viewed_food", number
            elif any(
                k in input_text
                for k in ["popular merchants", "top merchants", "famous merchants"]
            ):
                return "popular_merchant", number
            elif any(
                k in input_text
                for k in [
                    "exposure rate",
                    "view interpretation",
                    "interpret exposure",
                    "exposure",
                ]
            ):
                return "exposure_rate", None
            elif any(
                k in input_text
                for k in [
                    "food type",
                    "type of food",
                    "food category",
                    "category of food",
                ]
            ):
                return "food_type", None
            elif any(
                k in input_text
                for k in ["daily sales", "analyze sales", "sales today", "sales"]
            ):
                return "daily_sales", number
            elif any(
                k in input_text
                for k in [
                    "most ordered food",
                    "top orders",
                    "frequently ordered",
                    "most sales food",
                ]
            ):
                return "most_ordered_food", number

            return None, None

        # Main Chat UI
        def promptUser():
            st.title("📊 GAI 4.0o")
            gemini_ready = setup_gemini()

            st.subheader("💬 Ask me anything!")
            user_input = st.text_input("Your question:")

            # Session states
            for key in ["action", "awaiting_number", "user_name", "chat_history"]:
                if key not in st.session_state:
                    st.session_state[key] = [] if key == "chat_history" else None

            if user_input:
                action, num = extract_action_and_number(user_input)
                st.session_state["action"] = action

                if action:
                    if num is not None:
                        if isinstance(num, int) and num > 0:
                            if action == "daily_sales":
                                analyzeDailySales(merchantID, todayDate, num)
                            elif action == "most_viewed_food":
                                viewMostViewedFood(merchantID, num)
                            elif action == "popular_merchant":
                                popularMerchant(location, num)
                            elif action == "most_ordered_food":
                                analyzeMostOrderedFood(merchantID, todayDate, num)
                            st.session_state["action"] = None
                            st.session_state["awaiting_number"] = False
                        else:
                            st.error("Please enter a valid number greater than 0.")
                    else:
                        if action in [
                            "most_viewed_food",
                            "popular_merchant",
                            "daily_sales",
                            "most_ordered_food",
                        ]:
                            st.session_state["awaiting_number"] = True
                            st.info("How many results would you like to see?")
                        else:
                            if action == "exposure_rate":
                                viewInterpretation(merchantID)
                            elif action == "food_type":
                                analyzeFoodType(merchantID, location)
                            st.session_state["action"] = None
                elif not st.session_state["awaiting_number"]:
                    if gemini_ready:
                        gemini_response = get_gemini_response(user_input)
                        st.session_state["chat_history"].append(
                            {"user": user_input, "bot": gemini_response}
                        )
                        st.info(gemini_response)
                    else:
                        st.info(
                            "🔎 Try asking things like:\n- 'Show me top 5 foods'\n- 'Top merchants in my area'\n- 'Analyze today's sales'"
                        )

            # Prompt for missing number
            if st.session_state["awaiting_number"]:
                num_input = st.text_input("Enter number of results:")
                if num_input:
                    num = convert_to_number(num_input)
                    if isinstance(num, int) and num > 0:
                        if st.session_state["action"] == "most_viewed_food":
                            viewMostViewedFood(merchantID, num)
                        elif st.session_state["action"] == "popular_merchant":
                            popularMerchant(location, num)
                        elif st.session_state["action"] == "daily_sales":
                            analyzeDailySales(merchantID, todayDate, num)
                        elif st.session_state["action"] == "most_ordered_food":
                            analyzeMostOrderedFood(merchantID, todayDate, num)
                        st.session_state["awaiting_number"] = False
                        st.session_state["action"] = None
                    else:
                        st.error("Oops! I need a positive number. Try again?")

            # Show chat history
            if st.session_state["chat_history"] and gemini_ready:
                st.subheader("Chat History")
                for chat in st.session_state["chat_history"][-5:]:
                    st.text(f"You: {chat['user']}")
                    st.text(f"GAI: {chat['bot']}")
                    st.markdown("---")

        if __name__ == "__main__":
            promptUser()
