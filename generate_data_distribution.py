#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
import re

# Configure plot style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def count_images_in_directories(base_path):
    """Count images in each category directory."""
    category_counts = {}
    
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist. Please check the dataset path.")
        return {}
    
    # Walk through all subdirectories
    for category in sorted(os.listdir(base_path)):
        category_path = os.path.join(base_path, category)
        
        # Skip non-directories or hidden directories
        if not os.path.isdir(category_path) or category.startswith('.'):
            continue
        
        # Count image files in this category
        image_count = 0
        for file in os.listdir(category_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
        
        category_counts[category] = image_count
    
    return category_counts

def group_similar_categories(category_counts):
    """Group similar categories like different types of apples."""
    grouped_counts = {}
    
    for category, count in category_counts.items():
        # Extract the main fruit name (e.g., "Apple" from "Apple Red 1")
        main_fruit = re.split(r'\s+', category)[0]
        
        if main_fruit in grouped_counts:
            grouped_counts[main_fruit] += count
        else:
            grouped_counts[main_fruit] = count
    
    return grouped_counts

def plot_distribution(data, title, output_file, top_n=None, group_small=True):
    """Generate a bar chart of fruit distribution."""
    # Sort data by count (descending)
    sorted_data = OrderedDict(sorted(data.items(), key=lambda x: x[1], reverse=True))
    
    if top_n and len(sorted_data) > top_n:
        # Extract top N categories
        top_categories = list(sorted_data.items())[:top_n]
        
        if group_small:
            # Sum the counts of remaining categories
            others_sum = sum([count for _, count in list(sorted_data.items())[top_n:]])
            top_categories.append(("Others", others_sum))
    else:
        top_categories = list(sorted_data.items())
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get categories and counts
    categories = [item[0] for item in top_categories]
    counts = [item[1] for item in top_categories]
    
    # Create color gradient based on count values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(categories)))
    
    # Create the bars
    bars = ax.bar(categories, counts, color=colors, width=0.6, edgecolor='gray', linewidth=0.5)
    
    # Add count values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize the plot
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Fruit Categories', fontsize=14, labelpad=10)
    ax.set_ylabel('Number of Images', fontsize=14, labelpad=10)
    
    # Add a light grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add a thousand separator to y-axis
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Tight layout to make sure everything fits
    plt.tight_layout()
    
    # Add a subtle background color
    fig.patch.set_facecolor('#f8f9fa')
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Display the plot
    plt.close()

def main():
    # Paths for training and testing data
    training_path = 'data/fruits/Fruit-Images-Dataset-master/Training'
    test_path = 'data/fruits/Fruit-Images-Dataset-master/Test'
    
    print("Counting training images...")
    train_counts = count_images_in_directories(training_path)
    
    print("Counting test images...")
    test_counts = count_images_in_directories(test_path)
    
    # Combine training and test counts
    total_counts = {}
    for category in set(list(train_counts.keys()) + list(test_counts.keys())):
        total_counts[category] = train_counts.get(category, 0) + test_counts.get(category, 0)
    
    # Generate detailed distribution plot (all categories)
    plot_distribution(
        total_counts, 
        'Distribution of Images Across All Fruit Categories',
        'data_distribution_detailed.png'
    )
    
    # Generate summary distribution plot (top 20 categories)
    plot_distribution(
        total_counts, 
        'Distribution of Images Across Top 20 Fruit Categories',
        'data_distribution_top20.png',
        top_n=20
    )
    
    # Generate grouped distribution (main fruit types)
    grouped_counts = group_similar_categories(total_counts)
    plot_distribution(
        grouped_counts, 
        'Distribution of Images by Main Fruit Type',
        'data_distribution_grouped.png',
        top_n=20
    )
    
    print("Data distribution visualization complete!")

if __name__ == "__main__":
    main() 