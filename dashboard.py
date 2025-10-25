import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Belmont North Real Estate Insights",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess data
@st.cache_data
def load_data():
    # In a real scenario, you would load from file
    with open('data.json', 'r') as file:
        data = json.load(file)
    
    # Using the provided JSON data
    # data = {
    #     "results": [
    #         # ... (the full JSON data from your example)
    #         # Copy the entire JSON content here
    #     ]
    # }
    
    df = pd.DataFrame(data['results'])
    
    # Extract nested attributes
    bedrooms, bathrooms, garage_spaces, land_sizes, descriptions = [], [], [], [], []
    
    for attr in df['attributes']:
        bedrooms.append(attr.get('bedrooms', None))
        bathrooms.append(attr.get('bathrooms', None))
        garage_spaces.append(attr.get('garage_spaces', None))
        
        land_size = attr.get('land_size', None)
        if land_size and land_size != 'None' and land_size != 'nan':
            if isinstance(land_size, str):
                numeric_part = re.findall(r'\d+\.?\d*', land_size)
                land_sizes.append(float(numeric_part[0]) if numeric_part else None)
            else:
                land_sizes.append(float(land_size))
        else:
            land_sizes.append(None)
            
        descriptions.append(attr.get('description', ''))
    
    df['bedrooms'] = bedrooms
    df['bathrooms'] = bathrooms
    df['garage_spaces'] = garage_spaces
    df['land_size'] = land_sizes
    df['description'] = descriptions
    df['listing_date'] = pd.to_datetime(df['listing_date'])
    
    # Extract street name
    df['street_name'] = df['address'].apply(lambda x: x['street'])
    
    # Calculate price per bedroom
    df['price_per_bedroom'] = df['price'] / df['bedrooms']
    
    return df

# Load data
df = load_data()

# Dashboard Title
st.title("🏠 Belmont North Real Estate Market Analysis")
st.markdown("---")

# Key Metrics Row
st.subheader("📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_properties = len(df)
    st.metric("Total Properties", total_properties)

with col2:
    avg_price = df['price'].mean()
    st.metric("Average Price", f"${avg_price:,.0f}")

with col3:
    median_price = df['price'].median()
    st.metric("Median Price", f"${median_price:,.0f}")

with col4:
    total_value = df['price'].sum()
    st.metric("Total Market Value", f"${total_value:,.0f}")

# Second row of metrics
col5, col6, col7, col8 = st.columns(4)

with col5:
    avg_bedrooms = df['bedrooms'].mean()
    st.metric("Avg Bedrooms", f"{avg_bedrooms:.1f}")

with col6:
    avg_bathrooms = df['bathrooms'].mean()
    st.metric("Avg Bathrooms", f"{avg_bathrooms:.1f}")

with col7:
    houses = len(df[df['property_type'] == 'House'])
    st.metric("Houses", houses)

with col8:
    units = len(df[df['property_type'] == 'Unit'])
    st.metric("Units", units)

st.markdown("---")

# Main Content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Market Trends", "🏡 Property Types", "📍 Location Analysis", "💰 Price Analysis", "📋 Property Listings"])

with tab1:
    st.subheader("Market Trends & Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig_price_dist = px.histogram(df, x='price', nbins=20, 
                                    title='Price Distribution',
                                    labels={'price': 'Price (AUD)'})
        fig_price_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with col2:
        # Listing timeline
        monthly_listings = df.groupby(df['listing_date'].dt.to_period('M')).size().reset_index()
        monthly_listings['listing_date'] = monthly_listings['listing_date'].astype(str)
        
        fig_timeline = px.line(monthly_listings, x='listing_date', y=0,
                             title='Listings Timeline',
                             labels={'listing_date': 'Month', '0': 'Number of Listings'})
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Bedroom vs Price scatter
    st.subheader("Property Features vs Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bedroom_price = px.scatter(df, x='bedrooms', y='price', 
                                     color='property_type',
                                     title='Bedrooms vs Price',
                                     labels={'bedrooms': 'Number of Bedrooms', 'price': 'Price (AUD)'})
        st.plotly_chart(fig_bedroom_price, use_container_width=True)
    
    with col2:
        fig_bathroom_price = px.scatter(df, x='bathrooms', y='price',
                                      color='property_type',
                                      title='Bathrooms vs Price',
                                      labels={'bathrooms': 'Number of Bathrooms', 'price': 'Price (AUD)'})
        st.plotly_chart(fig_bathroom_price, use_container_width=True)

with tab2:
    st.subheader("Property Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Property type distribution
        prop_type_counts = df['property_type'].value_counts()
        fig_prop_type = px.pie(values=prop_type_counts.values, 
                             names=prop_type_counts.index,
                             title='Property Type Distribution')
        st.plotly_chart(fig_prop_type, use_container_width=True)
    
    with col2:
        # Average price by property type
        price_by_type = df.groupby('property_type')['price'].mean().reset_index()
        fig_avg_price_type = px.bar(price_by_type, x='property_type', y='price',
                                  title='Average Price by Property Type',
                                  labels={'property_type': 'Property Type', 'price': 'Average Price (AUD)'})
        st.plotly_chart(fig_avg_price_type, use_container_width=True)
    
    # Feature comparison
    st.subheader("Feature Comparison")
    
    feature_stats = df.groupby('property_type').agg({
        'bedrooms': 'mean',
        'bathrooms': 'mean', 
        'garage_spaces': 'mean',
        'price': 'mean'
    }).reset_index()
    
    fig_features = make_subplots(rows=2, cols=2, 
                               subplot_titles=('Avg Bedrooms', 'Avg Bathrooms', 'Avg Garage Spaces', 'Avg Price'))
    
    fig_features.add_trace(go.Bar(x=feature_stats['property_type'], y=feature_stats['bedrooms']), row=1, col=1)
    fig_features.add_trace(go.Bar(x=feature_stats['property_type'], y=feature_stats['bathrooms']), row=1, col=2)
    fig_features.add_trace(go.Bar(x=feature_stats['property_type'], y=feature_stats['garage_spaces']), row=2, col=1)
    fig_features.add_trace(go.Bar(x=feature_stats['property_type'], y=feature_stats['price']), row=2, col=2)
    
    fig_features.update_layout(height=600, showlegend=False, title_text="Property Features by Type")
    st.plotly_chart(fig_features, use_container_width=True)

with tab3:
    st.subheader("Geographical & Location Analysis")
    
    # Map visualization
    st.subheader("Property Locations")
    
    # Create map
    fig_map = px.scatter_mapbox(df, 
                              lat=df['coordinates'].apply(lambda x: x['latitude']),
                              lon=df['coordinates'].apply(lambda x: x['longitude']),
                              color='price',
                              size='price',
                              hover_name='area_name',
                              hover_data={'price': True, 'bedrooms': True, 'bathrooms': True},
                              title='Property Locations Map',
                              color_continuous_scale=px.colors.cyclical.IceFire,
                              zoom=12)
    
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Street analysis
    col1, col2 = st.columns(2)
    
    with col1:
        street_counts = df['street_name'].value_counts().head(10)
        fig_streets = px.bar(x=street_counts.values, y=street_counts.index,
                           orientation='h',
                           title='Most Common Streets',
                           labels={'x': 'Number of Properties', 'y': 'Street'})
        st.plotly_chart(fig_streets, use_container_width=True)
    
    with col2:
        # Price range by street
        street_prices = df.groupby('street_name').agg({
            'price': ['mean', 'count']
        }).round(0)
        street_prices.columns = ['avg_price', 'count']
        street_prices = street_prices[street_prices['count'] > 1].sort_values('avg_price', ascending=False).head(10)
        
        if not street_prices.empty:
            fig_street_prices = px.bar(street_prices, x='avg_price', y=street_prices.index,
                                     orientation='h',
                                     title='Average Price by Street (Multiple Listings)',
                                     labels={'avg_price': 'Average Price (AUD)', 'y': 'Street'})
            st.plotly_chart(fig_street_prices, use_container_width=True)

with tab4:
    st.subheader("Detailed Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price ranges
        price_ranges = ['<$700K', '$700K-$1M', '$1M-$1.5M', '$1.5M-$2M', '>$2M']
        price_bins = [0, 700000, 1000000, 1500000, 2000000, float('inf')]
        df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_ranges)
        
        range_counts = df['price_range'].value_counts()
        fig_price_ranges = px.pie(values=range_counts.values, 
                                names=range_counts.index,
                                title='Price Range Distribution')
        st.plotly_chart(fig_price_ranges, use_container_width=True)
    
    with col2:
        # Price per bedroom
        fig_ppb = px.box(df, y='price_per_bedroom', 
                       title='Price per Bedroom Distribution',
                       labels={'price_per_bedroom': 'Price per Bedroom (AUD)'})
        st.plotly_chart(fig_ppb, use_container_width=True)
    
    # High-end vs Budget analysis
    st.subheader("Market Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏰 High-End Properties (>$1.5M)**")
        high_end = df[df['price'] > 1500000]
        if not high_end.empty:
            for _, prop in high_end.iterrows():
                st.write(f"• **{prop['area_name']}** - ${prop['price']:,} ({prop['bedrooms']} bed, {prop['bathrooms']} bath)")
        else:
            st.write("No high-end properties in current data")
    
    with col2:
        st.markdown("**💰 Budget Properties (<$700K)**")
        budget = df[df['price'] < 700000]
        if not budget.empty:
            for _, prop in budget.iterrows():
                st.write(f"• **{prop['area_name']}** - ${prop['price']:,} ({prop['property_type']})")
        else:
            st.write("No budget properties in current data")
    
    # Land size vs Price
    st.subheader("Land Size Analysis")
    valid_land = df[df['land_size'].notna()]
    if not valid_land.empty:
        fig_land_price = px.scatter(valid_land, x='land_size', y='price',
                                  color='property_type',
                                  title='Land Size vs Price',
                                  labels={'land_size': 'Land Size (m²)', 'price': 'Price (AUD)'},
                                  trendline="lowess")
        st.plotly_chart(fig_land_price, use_container_width=True)

with tab5:
    st.subheader("Property Listings")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_range = st.slider("Price Range (AUD)", 
                              min_value=int(df['price'].min()), 
                              max_value=int(df['price'].max()),
                              value=(int(df['price'].min()), int(df['price'].max())))
    
    with col2:
        bedroom_filter = st.selectbox("Bedrooms", 
                                    options=["All"] + sorted(df['bedrooms'].dropna().unique().astype(int).tolist()))
    
    with col3:
        property_type_filter = st.selectbox("Property Type", 
                                          options=["All"] + df['property_type'].unique().tolist())
    
    with col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        reset_filters = st.button("Reset Filters")
    
    # Apply filters
    filtered_df = df.copy()
    
    if price_range:
        filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                                (filtered_df['price'] <= price_range[1])]
    
    if bedroom_filter != "All":
        filtered_df = filtered_df[filtered_df['bedrooms'] == bedroom_filter]
    
    if property_type_filter != "All":
        filtered_df = filtered_df[filtered_df['property_type'] == property_type_filter]
    
    # Display filtered results
    st.write(f"**Found {len(filtered_df)} properties**")
    
    for _, property in filtered_df.iterrows():
        with st.expander(f"🏠 {property['area_name']} - ${property['price']:,}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Property Type:** {property['property_type']}")
                st.write(f"**Bedrooms:** {property['bedrooms']} | **Bathrooms:** {property['bathrooms']} | **Garage:** {property['garage_spaces']}")
                if property['land_size']:
                    st.write(f"**Land Size:** {property['land_size']} m²")
                st.write(f"**Listing Date:** {property['listing_date'].strftime('%Y-%m-%d')}")
                
                # Show first 200 characters of description
                description = property['description']
                if len(description) > 200:
                    description = description[:200] + "..."
                st.write(f"**Description:** {description}")
            
            with col2:
                st.write(f"**Price:** ${property['price']:,}")
                st.write(f"**Price/Bedroom:** ${property['price_per_bedroom']:,.0f}")
                st.write(f"**Coordinates:** {property['coordinates']['latitude']:.4f}, {property['coordinates']['longitude']:.4f}")

# Sidebar with additional insights
with st.sidebar:
    st.header("🔍 Quick Insights")
    
    st.subheader("Top 5 Most Expensive")
    top_expensive = df.nlargest(5, 'price')[['area_name', 'price']]
    for _, prop in top_expensive.iterrows():
        st.write(f"• {prop['area_name']}")
        st.write(f"  ${prop['price']:,}")
    
    st.subheader("Best Value (Lowest Price/Bedroom)")
    best_value = df.nsmallest(3, 'price_per_bedroom')[['area_name', 'price_per_bedroom', 'bedrooms']]
    for _, prop in best_value.iterrows():
        st.write(f"• {prop['area_name']}")
        st.write(f"  ${prop['price_per_bedroom']:,.0f}/bedroom ({prop['bedrooms']} beds)")
    
    st.subheader("Feature Frequency")
    descriptions = ' '.join(df['description'].astype(str)).lower()
    features = {
        'air conditioning': 'air conditioning',
        'renovated': 'renovated|refurbished',
        'solar': 'solar',
        'pool': 'pool',
        'view': 'view'
    }
    
    for feature, pattern in features.items():
        count = len(re.findall(pattern, descriptions))
        st.write(f"• {feature.title()}: {count} mentions")
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the filters in the Property Listings tab to find properties matching your criteria.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Real Estate Market Analysis Dashboard | Belmont North, NSW | Data updated automatically"
    "</div>",
    unsafe_allow_html=True
)