import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
from babel.numbers import format_currency

# Set daily_orders function to return daily_orders_df
def daily_orders(df):
    daily_orders_df = df.resample(rule='D', on='order_date').agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value','sum')
        ).reset_index()
    
    return daily_orders_df

# Set order_product_category function to return order_by_product_category_df
def order_product_category(df):
    order_by_product_category_df = df.groupby(by="product_category").agg(
        num_of_order = ('order_id','count'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return order_by_product_category_df

# Set count_customers function to return customers_in_cities and customers_in_states
def count_customers(df):
    customers_in_cities = df.groupby(by="customer_city").agg(
        count_customer = ('customer_unique_id','nunique')
        ).reset_index()
    
    customers_in_states = df.groupby(by="customer_state").agg(
        count_customer = ('customer_unique_id','nunique')
        ).reset_index()
    
    return customers_in_cities, customers_in_states

# Set customers_order function to return count_sum_order
def customers_order(df):
    cust_count_sum_order = df.groupby(by="customer_unique_id").agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return cust_count_sum_order

# Set count_sellers function to return sellers_in_cities and sellers_in_states
def count_sellers(df):
    sellers_in_cities = df.groupby(by="seller_city").agg(
        count_seller = ('seller_id','nunique')
        ).reset_index()
    
    sellers_in_states = df.groupby(by="seller_state").agg(
        count_seller = ('seller_id','nunique')
        ).reset_index()
    
    return sellers_in_cities, sellers_in_states

# Set customers_order function to return count_sum_order
def sellers_order(df):
    seller_count_sum_order = df.groupby(by="seller_id").agg(
        count_order = ('order_id','nunique'), 
        sum_order_value = ('total_order_value', 'sum')
        ).reset_index()
    
    return seller_count_sum_order

#Set palette colors
colors=["#3187d4",'#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4','#b3bcc4']

#Load data from cleaned dataframe
main_df = pd.read_csv('C:\projek akhir analisis data\main_data_for_dashboard.csv')

#Looping for change column to datetime type
dt_columns = ['order_date', 'approved_date', 'shipped_date', 'delivery_date']
main_df.sort_values(by="order_date", inplace=True)
main_df.reset_index(inplace=True)

for column in dt_columns:
    main_df[column] = pd.to_datetime(main_df[column])

#Set min_date and max_date for filter data
min_date = main_df["order_date"].min()
max_date = main_df["order_date"].max()

#Add sidebar
with st.sidebar:
    #Add company brand
    st.image("https://seeklogo.com/images/O/olist-logo-9DCE4443F8-seeklogo.com.png")
    
    #Make start_date & end_date from date_input
    start_date, end_date = st.date_input(
        label='Date Range', 
        min_value=min_date,
        max_value=max_date,
        value= [min_date, max_date]
        )

#Add one day to end_date and subs
start_date = start_date - pd.DateOffset(days=1)
end_date = end_date + pd.DateOffset(days=1)

#Filtered main_df by start_date and end_date
main_df = main_df[(main_df["order_date"] >= start_date) & 
                (main_df["order_date"] <= end_date)]

# Make the title dashboard
st.markdown('<h1 style="text-align: center;">E-Commerce Olist Dashboard</h1>', unsafe_allow_html=True)

################################### ORDERS ###################################
def orders_analysis():
    daily_orders_df = daily_orders(main_df)
    order_by_product_category_df = order_product_category(main_df)

    #Count Orders and Total Order Value per Day
    st.subheader("Daily Orders")

    col1, col2 = st.columns(2)
    with col1:
        total_orders = daily_orders_df.count_order.sum()
        st.metric("Total Orders", value=total_orders)
 
    with col2:
        total_order_value = format_currency(daily_orders_df.sum_order_value.sum(), "R$", locale='pt_BR') 
        st.metric("Total Order Value", value=total_order_value)
    
    #Set max value
    xmax = daily_orders_df.order_date[np.argmax(daily_orders_df.count_order)]
    ymax = daily_orders_df.count_order.max()

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(daily_orders_df["order_date"],
            daily_orders_df["count_order"],
            marker='o', 
            linewidth=3,
            color= "#3187d4"
            )
    ax.set_title("Number of Order per Day", loc="center", fontsize=30, pad=20)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.annotate(f"At {xmax.strftime('%Y-%m-%d')}\n have {ymax} orders", 
                xy=(xmax, ymax), xycoords='data',
                xytext=(xmax + (end_date - start_date)/6, ymax), #Give annotate with 1:6 scale of x
                textcoords='data', size=20, va="center", ha="center",
                bbox=dict(boxstyle="round4", fc="w"),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", fc="w")
                )
    st.pyplot(fig)

    #Set max value
    x_max = daily_orders_df.order_date[np.argmax(daily_orders_df.sum_order_value)]
    y_max = round(daily_orders_df.sum_order_value.max(), 2)
    
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.plot(daily_orders_df["order_date"],
            daily_orders_df["sum_order_value"],
            marker='o', 
            linewidth=3,
            color= "#3187d4"
            )
    ax.set_title("Total Order Value per Day", loc="center", fontsize=30, pad=20)
    ax.set_ylabel("R$", fontsize=20, labelpad=10)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.annotate(f"At {x_max.strftime('%Y-%m-%d')}\n have value\n R$ {y_max}", 
            xy=(x_max, y_max), xycoords='data',
            xytext=(x_max + (end_date - start_date)/6, y_max), #Give annotate with 1:6 scale of x
            textcoords='data', size=20, va="center", ha="center",
            bbox=dict(boxstyle="round4", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", fc="w")
            )
    st.pyplot(fig)

    #Best and Worst Performing Product Category
    st.subheader("Best and Worst Performing Product Category")
 
    tab1, tab2 = st.tabs(['Count Order', 'Total Order Value'])
    
    with tab1:
        col1, col2 = st.columns(2)
 
        with col2:
            min_order = order_by_product_category_df.num_of_order.min()
            st.metric("Lowest Number of Order by Product Category", value=min_order)
 
        with col1:
            max_order = order_by_product_category_df.num_of_order.max()
            st.metric("Highest Number of Order by Product Category", value=max_order)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

        sns.barplot(
            x="num_of_order",
            y="product_category",
            data= order_by_product_category_df.sort_values('num_of_order', ascending=False).head(10),
            palette= colors,
            ax=ax[0]
            )
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Number of Order', fontsize=15, labelpad=10)
        ax[0].set_title("Highest Number Ordered", loc="center", fontsize=20, pad=10)
        ax[0].tick_params(axis ='y', labelsize=18)
        ax[0].tick_params(axis ='x', labelsize=18)

        sns.barplot(
            x="num_of_order",
            y="product_category",
            data= order_by_product_category_df.sort_values(by=['num_of_order','sum_order_value'], ascending=True).head(10),
            palette=colors,
            ax=ax[1]
            )
        ax[1].set_ylabel(None)
        ax[1].set_xlabel('Number of Order', fontsize=15, labelpad=10)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_title("Lowest Number Ordered", loc="center", fontsize=20, pad=10)
        ax[1].tick_params(axis='y', labelsize=18)
        ax[1].tick_params(axis='x', labelsize=18)
        plt.suptitle("Best and Worst Performing Product Category by Number Ordered", fontsize=25)

        st.pyplot(fig)

    with tab2:
        col1, col2 = st.columns(2)
 
        with col1:
            max_order_value = format_currency(order_by_product_category_df.sum_order_value.max(), "R$", locale='pt_BR')
            st.metric("Highest Total Order Value by Product Category", value=max_order_value)

        with col2:
            min_order_value = format_currency(order_by_product_category_df.sum_order_value.min(), "R$", locale='pt_BR')
            st.metric("Lowest Total Order Value by Product Category", value=min_order_value)
 
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25,10))

        sns.barplot(
            x="sum_order_value",
            y="product_category",
            data= order_by_product_category_df.sort_values('sum_order_value', ascending=False).head(10),
            palette= colors,
            ax=ax[0]
            )
        ax[0].set_ylabel(None)
        ax[0].set_xlabel('Total Order Value (Million R$)', fontsize=15, labelpad=10)
        ax[0].set_title("Highest Total Order Value", loc="center", fontsize=20, pad=10)
        ax[0].tick_params(axis ='y', labelsize=18)
        ax[0].tick_params(axis ='x', labelsize=18)

        sns.barplot(
            x="sum_order_value",
            y="product_category",
            data= order_by_product_category_df.sort_values('sum_order_value', ascending=True).head(10),
            palette= colors,
            ax=ax[1]
            )
        ax[1].set_ylabel(None)
        ax[1].set_xlabel('Total Order Value (R$)', fontsize=15, labelpad=10)
        ax[1].invert_xaxis()
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_title("Lowest Total Order Value", loc="center", fontsize=20, pad=10)
        ax[1].tick_params(axis='y', labelsize=18)
        ax[1].tick_params(axis='x', labelsize=18)
        plt.suptitle("Best and Worst Performing Product Category by Total Order Value", fontsize=25)

        st.pyplot(fig)

################################### CUSTOMERS ###################################
def customers_analysis():
    customers_in_cities, customers_in_states = count_customers(main_df)
    cust_count_sum_order = customers_order(main_df)

    #Distribution of Customers by City and State
    st.subheader("Distribution of Customers by City and State")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_count_cust = main_df.customer_unique_id.nunique()
        st.metric("Total Number of Customers", value=total_count_cust)

    with col2:
        highest_count_cust_city = customers_in_cities.count_customer.max()
        st.metric("Highest by City", value=highest_count_cust_city)

    with col3:
        highest_count_cust_state = customers_in_states.count_customer.max()
        st.metric("Highest by State", value=highest_count_cust_state)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

    sns.barplot(x="customer_city", 
                y="count_customer", 
                data= customers_in_cities.sort_values('count_customer', ascending=False).head(10), 
                palette= colors, 
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_title("Based on City", loc="center", fontsize=18, pad=10)
    ax[0].tick_params(axis ='y', labelsize=15)
    ax[0].tick_params(axis ='x', labelsize=15)

    sns.barplot(x="customer_state", 
                y="count_customer", 
                data= customers_in_states.sort_values('count_customer', ascending=False).head(10),
                palette= colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_title("Based on State", loc="center", fontsize=18, pad=10)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis ='x', labelsize=15)

    plt.suptitle("Distribution of Number of Customers by City and State", fontsize=20)
    st.pyplot(fig)

    #Customer with Largest Order
    st.subheader("Customer with Largest Number of Order and Total Order Value")
    tab1, tab2 = st.tabs(['Count Order','Total Order Value'])
 
    with tab1:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_cust_count_order = cust_count_sum_order.count_order.max()
            st.metric("Highest Number of Order", value=max_cust_count_order)

        with col2:
            min_cust_count_order = cust_count_sum_order.count_order.min()
            st.metric("Lowest Number of Order", value=min_cust_count_order)

        with col3:
            avg_cust_count_order = cust_count_sum_order.count_order.mean().astype(int)
            st.metric("Average Number of Order", value=avg_cust_count_order)
        
        fig, ax = plt.subplots(figsize=(25, 10))
        sns.barplot(x="count_order", 
                y="customer_unique_id", 
                data= cust_count_sum_order.sort_values('count_order',ascending=False).head(10), 
                palette= colors,
                ax=ax)
        ax.set_ylabel('Customer Unique ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Number of Order', fontsize=18, labelpad=10)
        ax.set_title("Customer with Largest Number of Order", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)
 
    with tab2:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_cust_order_value = format_currency(cust_count_sum_order.sum_order_value.max(), "R$", locale='pt_BR')
            st.metric("Highest Total Order Value", value=max_cust_order_value)

        with col2:
            min_cust_order_value = format_currency(cust_count_sum_order.sum_order_value.min(), "R$", locale='pt_BR')
            st.metric("Lowest Total Order Value", value=min_cust_order_value)
        
        with col3:
            avg_cust_order_value = format_currency(cust_count_sum_order.sum_order_value.mean(), "R$", locale='pt_BR')
            st.metric("Average Total Order Value", value=avg_cust_order_value)
        
        fig, ax = plt.subplots(figsize=(25, 10))

        sns.barplot(x="sum_order_value", 
                    y="customer_unique_id", 
                    data= cust_count_sum_order.sort_values('sum_order_value',ascending=False).head(10), 
                    palette= colors,
                    ax=ax)
        ax.set_ylabel('Customer Unique ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Total Order Value (R$)', fontsize=18, labelpad=10)
        ax.set_title("Customer with Largest Total Order Value", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)

            
################################### SELLERS ###################################
def sellers_analysis():
    sellers_in_cities, sellers_in_states = count_sellers(main_df)
    seller_count_sum_order = sellers_order(main_df)

    #Distribution of Sellers by City and State
    st.subheader("Distribution of Sellers by City and State")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_count_seller = main_df.seller_id.nunique()
        st.metric("Total Number of Sellers", value=total_count_seller)

    with col2:
        highest_count_seller_city = sellers_in_cities.count_seller.max()
        st.metric("Highest by City", value=highest_count_seller_city)

    with col3:
        highest_count_seller_state = sellers_in_states.count_seller.max()
        st.metric("Highest by State", value=highest_count_seller_state)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

    sns.barplot(x="seller_city", 
                y="count_seller", 
                data= sellers_in_cities.sort_values('count_seller', ascending=False).head(10), 
                palette= colors, 
                ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_title("Based on City", loc="center", fontsize=18, pad=10)
    ax[0].tick_params(axis ='y', labelsize=15)
    ax[0].tick_params(axis='x', labelsize=15)

    sns.barplot(x="seller_state", 
                y="count_seller", 
                data= sellers_in_states.sort_values('count_seller', ascending=False).head(10),
                palette= colors, 
                ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].tick_params(axis='x', labelrotation=50)
    ax[1].set_title("Based on State", loc="center", fontsize=18, pad=10)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=15)

    plt.suptitle("Distribution of Number of Sellers by City and State", fontsize=20)
    st.pyplot(fig)

    #Seller with Largest Order
    st.subheader("Seller with Largest Number of Order and Total Order Value")
    tab1, tab2 = st.tabs(['Count Order','Total Order Value'])
 
    with tab1:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_seller_count_order = seller_count_sum_order.count_order.max()
            st.metric("Highest Number of Order", value=max_seller_count_order)

        with col2:
            min_seller_count_order = seller_count_sum_order.count_order.min()
            st.metric("Lowest Number of Order", value=min_seller_count_order)

        with col3:
            avg_seller_count_order = seller_count_sum_order.count_order.mean().astype(int)
            st.metric("Average Number of Order", value=avg_seller_count_order)
        
        fig, ax = plt.subplots(figsize=(25, 10))
        sns.barplot(x="count_order", 
                    y="seller_id", 
                    data= seller_count_sum_order.sort_values('count_order',ascending=False).head(10), 
                    palette= colors,
                    ax=ax)
        ax.set_ylabel('Seller ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Number of Order', fontsize=18, labelpad=10)
        ax.set_title("Seller with Largest Number of Order", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)
 
    with tab2:
        col1, col2, col3 = st.columns(3)
 
        with col1:
            max_seller_order_value = format_currency(seller_count_sum_order.sum_order_value.max(), "R$", locale='pt_BR')
            st.metric("Highest Total Order Value", value=max_seller_order_value)

        with col2:
            min_seller_order_value = format_currency(seller_count_sum_order.sum_order_value.min(), "R$", locale='pt_BR')
            st.metric("Lowest Total Order Value", value=min_seller_order_value)
        
        with col3:
            avg_seller_order_value = format_currency(seller_count_sum_order.sum_order_value.mean(), "R$", locale='pt_BR')
            st.metric("Average Total Order Value", value=avg_seller_order_value)

        
        fig, ax = plt.subplots(figsize=(25, 10))

        sns.barplot(x="sum_order_value", 
                    y="seller_id", 
                    data= seller_count_sum_order.sort_values('sum_order_value',ascending=False).head(10), 
                    palette= colors,
                    ax=ax)
        ax.set_ylabel('Seller ID', fontsize=18, labelpad=10)
        ax.set_xlabel('Total Order Value (R$)', fontsize=18, labelpad=10)
        ax.set_title("Seller with Largest Total Order Value", loc="center", fontsize=20, pad=10)
        ax.bar_label(ax.containers[0], label_type='center')
        ax.tick_params(axis ='y', labelsize=15)
        ax.tick_params(axis ='x', labelsize=15)
        st.pyplot(fig)

#Make function with radio button on sidebar to call analysis function
def sidebar_function():
    with st.sidebar:
        selected= option_menu(
            menu_title= "Analyze What?",
            options=["Orders","Customers","Sellers"],
            icons=["cart-fill","people-fill","shop-window"],
            menu_icon="clipboard-data-fill",
            default_index=0
            )

    if selected =="Orders":
        orders_analysis()
    if selected=="Customers":
        customers_analysis()
    if selected=="Sellers":
        sellers_analysis()
sidebar_function()

st.sidebar.caption('Copyright © Fakhriyunansah - 2024')