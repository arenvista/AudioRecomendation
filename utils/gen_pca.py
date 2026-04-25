import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D  # <-- Imported to build custom legend handles

import matplotlib.colors as mcolors # <-- New import needed for color conversion

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

def highlight_specific_value(df, target_col, target_value, assign_color='red'):
    """
    Checks a column for a specific value. If found, assigns the highlight color
    to the 'Point_Color' column in RGBA format without triggering shape errors.
    """
    if target_col not in df.columns:
        print(f"Warning: '{target_col}' not found.")
        return df, {}

    # 1. Convert string colors into RGBA tuples
    rgba_assign_color = mcolors.to_rgba(assign_color)
    rgba_base_color = mcolors.to_rgba("lightgrey")

    # 2. Safety check: Initialize Point_Color with the base color if it doesn't exist
    if "Point_Color" not in df.columns:
        df["Point_Color"] = [rgba_base_color for _ in range(len(df))]

    # 3. Update Point_Color using a list comprehension
    # This loops through both the target column and the current colors simultaneously.
    # If the value matches, it injects the new color. Otherwise, it keeps the current color.
    df["Point_Color"] = [
        rgba_assign_color if val == target_value else current_color
        for val, current_color in zip(df[target_col], df["Point_Color"])
    ]
    
    # 4. Build a simple dictionary for the legend
    color_dict = {
        target_value: rgba_assign_color,
        "Other": rgba_base_color 
    }
    
    return df, color_dict

def get_column_colors(df, column_name, cmap_name='tab10', na_color='lightgrey', na_label='Unknown'):
    """
    Generates a unique color for each unique value in a DataFrame column,
    assigning a specific fallback color to missing (NaN) values.
    
    Args:
        df: The pandas DataFrame.
        column_name: The name of the column to generate colors for.
        cmap_name: The matplotlib colormap to use.
        na_color: The specific color to use for missing values (default: 'lightgrey').
        na_label: The string to use in the legend for missing values (default: 'Unknown').
        
    Returns:
        color_dict: A dictionary mapping {unique_value: rgba_color}
        color_series: A pandas Series of colors corresponding to the rows in df
    """
    # 1. Work with a temporary column where NaNs are filled with our label
    filled_col = df[column_name].fillna(na_label)
    
    # 2. Find all unique valid values (ignoring NaNs for the gradient calculation)
    unique_values = df[column_name].dropna().unique()
    
    # 3. Load the specified matplotlib colormap
    cmap = plt.get_cmap(cmap_name)
    
    # 4. Generate evenly spaced colors across the colormap for valid data
    color_rgba_values = cmap(np.linspace(0, 1, len(unique_values)))
    
    # 5. Zip the valid unique values and colors together into a dictionary
    color_dict = dict(zip(unique_values, color_rgba_values))
    
    # 6. If there are actually any missing values, add the specific na_color to the dict
    if df[column_name].isna().any():
        # mcolors.to_rgba ensures 'lightgrey' becomes an RGBA array just like the colormap outputs
        color_dict[na_label] = mcolors.to_rgba(na_color)
        
    # 7. Map the colors back to the rows using our filled column
    color_series = filled_col.map(color_dict)
    
    return color_dict, color_series

def plot_pca_from_csv(csv_filepath, output_filepath=None, label="Label"):
    """
    Reads a PCA CSV file and regenerates the 3D scatter plot.
    """
    print(f"Loading data from: {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    print(df.head())
    
    # Setup 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
# --- Determine base coloring ---
    if label in df.columns:
        color_mapping, row_colors = get_column_colors(df, label, cmap_name='viridis')
        df['Point_Color'] = row_colors
    else:
        df['Point_Color'] = 'lightgrey'
        color_mapping = {}
        
    # --- HIGHLIGHT BEFORE PLOTTING ---
    df, hl_map1 = highlight_specific_value(
        df, target_col="Artist", target_value="Arctic Monkeys", assign_color='red'
    )
    df, hl_map2 = highlight_specific_value(
        df, target_col="Artist", target_value="Earth Wind and Fire", assign_color='orange'
    )

    # Update our legend mapping with the new highlighted colors
    if label in df.columns:
        color_mapping["Arctic Monkeys"] = hl_map1.get("Arctic Monkeys", color_mapping.get("Arctic Monkeys"))
        color_mapping["Earth Wind and Fire"] = hl_map2.get("Earth Wind and Fire", color_mapping.get("Earth Wind and Fire"))

    # --- Plot with 3 dimensions ---
    # NOW we draw the plot, after the dataframe has been updated with red/orange
    scatter = ax.scatter(
        df['PC1'], 
        df['PC2'], 
        df['PC3'], 
        c=df['Point_Color'], 
        s=50
    )


    # --- Create the legend ---
    if label in df.columns and color_mapping:
        # Build custom legend handles using the color_mapping dictionary
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=val)
            for val, color in color_mapping.items()
        ]
        
        # Add the legend to the axes
        ax.legend(handles=legend_handles, title=label, loc='center left', bbox_to_anchor=(1.05, 0.5))
    # ------------------------------
    
    print("Annotating PCA...")
    # for index, row in df.iterrows():
    #     # Add text in 3D space
    #     ax.text(
    #         row['PC1'], 
    #         row['PC2'], 
    #         row['PC3'], 
    #         row['Filename'], 
    #         fontsize=8
    #     )
        
    ax.set_title("Audio Feature Space (3D PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    
# Save or Show
    if output_filepath:
        output_path = Path(output_filepath)
        
        # --- NEW: Add bbox_inches='tight' to prevent the legend from being cut off ---
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2) 
        # ----------------------------------------------------------------------------
        
        print(f"Saved PCA Img => {output_path}")
        plt.close()
    else:
        # tight_layout helps when displaying via the GUI window as well
        plt.tight_layout() 
        plt.show()
    print(df["Artist"])
    print(df["Point_Color"])
    print(df.columns)
    return df

def plot_2d_pca_from_csv(csv_filepath, output_filepath=None, label="Label"):
    """
    Reads a PCA CSV file and generates a 2D scatter plot using PC1 and PC2.
    """
    print(f"Loading data for 2D plot from: {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    
    # Setup 2D figure (Notice we don't use projection='3d' here)
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # --- Determine base coloring ---
    if label in df.columns:
        color_mapping, row_colors = get_column_colors(df, label, cmap_name='viridis')
        df['Point_Color'] = row_colors
    else:
        # Fallback if the label column doesn't exist
        # We wrap the RGBA array in a list so Pandas assigns it correctly to every row
        rgba_base = mcolors.to_rgba('lightgrey')
        df['Point_Color'] = [rgba_base for _ in range(len(df))]
        color_mapping = {}
        
    # --- HIGHLIGHT BEFORE PLOTTING ---
    df, hl_map1 = highlight_specific_value(
        df, target_col="Artist", target_value="Arctic Monkeys", assign_color='red'
    )
    df, hl_map2 = highlight_specific_value(
        df, target_col="Artist", target_value="Earth Wind and Fire", assign_color='orange'
    )

    # Update our legend mapping with the new highlighted colors
    if label in df.columns:
        color_mapping["Arctic Monkeys"] = hl_map1.get("Arctic Monkeys", color_mapping.get("Arctic Monkeys"))
        color_mapping["Earth Wind and Fire"] = hl_map2.get("Earth Wind and Fire", color_mapping.get("Earth Wind and Fire"))

    # --- Plot with 2 dimensions ---
    # We only use PC1 and PC2 here!
    scatter = ax.scatter(
        df['PC1'], 
        df['PC2'], 
        c=df['Point_Color'], 
        s=50,
        alpha=0.8 # Adding slight transparency helps see overlapping points in 2D
    )

    # --- Create the legend ---
    if label in df.columns and color_mapping:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=val)
            for val, color in color_mapping.items()
        ]
        
        # Add the legend to the outside right
        ax.legend(handles=legend_handles, title=label, loc='center left', bbox_to_anchor=(1.05, 0.5))
    # ------------------------------
        
    ax.set_title("Audio Feature Space (2D PCA)")
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    
    # Add a faint grid to make the 2D space easier to read
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save or Show
    if output_filepath:
        output_path = Path(output_filepath)
        # Remember bbox_inches='tight' so the legend doesn't get cut off!
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2) 
        print(f"Saved 2D PCA Img => {output_path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    return df

if __name__ == "__main__":
    csv_in = "/home/sybil/Documents/School/2026-Spring/CMSC-475/Project/AudioRecomendation/output/pca/2026-04-09_09-02-54_pca.csv.augmented.csv"
    img_out = "/home/sybil/Documents/School/2026-Spring/CMSC-475/Project/AudioRecomendation/output/pca/regenerated_plot"
    img_out_2 = "/home/sybil/Documents/School/2026-Spring/CMSC-475/Project/AudioRecomendation/output/pca/regenerated_plot2D"
    
    fields = ["Artist", "Album", "Genre"]
    for f in fields:
        df = plot_pca_from_csv(csv_in, img_out+f"3D_{f}.png", f)
        df.to_csv(csv_in + ".colored.csv")
        df = plot_2d_pca_from_csv(csv_in, img_out_2+f"2D_{f}.png", f)
    # df.to_csv(csv_in + ".colored.csv")
