import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import couchdb
from main import PaddedFrame


class ViewReportsFrame(PaddedFrame):
    def __init__(self, master, parent=None, parent_frame=None, model=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.callback = callback
        self.parent = parent
        self.master = master

        # Create a frame for the checkboxes and buttons
        self.control_frame = tk.Frame(self)
        self.control_frame.grid(row=0, column=0, sticky='nsew')

        # Create a frame for the figure canvas and button
        self.canvas_frame = tk.Frame(self)

        # Connect to the CouchDB server
        couch = couchdb.Server('http://admin:1234@localhost:5984')

        # Select the database
        db = couch['test_dataframes']

        # Fetch all documents from the database
        rows = db.view('_all_docs', include_docs=True)

        # Extract the document data from each row
        self.docs = [row['doc'] for row in rows]

        # Extract all unique names from the documents
        self.names = set()
        for doc in self.docs:
            self.names.update(doc['Name'].values())
        self.names = list(self.names)

        # Create a mapping from names to numbers
        self.name_to_number = {name: i for i, name in enumerate(self.names)}

        # Create a list to store the checkboxes
        self.checkboxes = []
        self.checkbox_states = {}
        j = 0
        i = 0

        # Loop over the names
        for name in self.names:
            # Create a variable to store the state of the checkbox
            var = tk.BooleanVar()

            # Create a checkbox for the name
            checkbox = tk.Checkbutton(self.control_frame, text=self.name_to_number[name], variable=var)

            # Add the checkbox to the list
            self.checkboxes.append(checkbox)

            # Update the checkbox_states dictionary
            self.checkbox_states[name] = var

            if i > 7:
                # Display the checkbox
                i = 0
                j += 1
            checkbox.grid(row=i, column=j, sticky='w')
            i += 1

        # Back button
        self.back_button = ttk.Button(self.control_frame, text="Back", command=self.go_back)
        self.back_button.grid(row=8, column=0, columnspan=2)

        # Display Graphs button
        self.display_button = ttk.Button(self.control_frame, text="Display Graphs", command=self.create_plot)
        self.display_button.grid(row=9, column=0, columnspan=2)

    def get_data(self, name):
        # Initialize an empty DataFrame to store the data
        data = pd.DataFrame()

        # Loop over all documents
        for doc in self.docs:
            # Convert the document to a DataFrame
            df = pd.DataFrame(doc)

            # Filter the DataFrame based on the test subject name and session type
            filtered_df = df[(df['Name'] == name)]

            # If the filtered data is empty, skip this document
            if filtered_df.empty:
                continue

            # Append the filtered DataFrame to the data DataFrame
            data = pd.concat([data, filtered_df])

        # Drop duplicate rows based on 'Name' and 'Session Type'
        data = data.drop_duplicates(subset=['Name', 'Session Type'])

        return data

    def create_plot(self):
        # Create the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        # Create the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Create a frame for the back button
        button_frame = tk.Frame(self.canvas_frame)
        button_frame.grid(row=1, column=0, sticky='nsew')

        # Back button in the canvas frame
        back_button = ttk.Button(button_frame, text="Back", command=self.go_back)
        back_button.grid(row=1, column=1, sticky='nsew')

        # Hide the control frame and show the canvas frame
        self.control_frame.grid_remove()
        self.canvas_frame.grid(row=0, column=0, sticky='nsew')

        self.update_plot()

    def update_plot(self):
        # Clear the current plot
        self.ax.clear()

        # Define the colors for each metric
        colors = ['turquoise', 'deeppink', 'orangered', 'yellow']

        # Define the edge colors for each session type
        edge_colors = ['teal', 'darkorchid']

        # Define the metrics
        metrics = ['Response Accuracy', 'Commission Error', 'Omission Error', 'Mean Delay']

        # Get the filenames of the selected checkboxes
        selected_filenames = [checkbox.cget("text") for checkbox, var in
                              zip(self.checkboxes, self.checkbox_states.values()) if var.get()]

        # Extract all unique test subject names from the documents
        unique_names = set()
        for doc in self.docs:
            unique_names.update(doc['Name'].values())
        unique_names = list(unique_names)

        # Get the names of the selected checkboxes
        selected_names = [name for name, var in self.checkbox_states.items() if var.get()]
        print("selected_names = ", selected_names)
        # Define the total number of ticks
        total_ticks = len(selected_names)

        # Define the total number of session types
        total_session_types = 2

        # Define the total number of bars per session type
        total_bars_per_session = len(metrics)

        # Define the bar width
        bar_width = 0.35

        # Define the gap between groups of bars
        gap = 0.2

        # Calculate the width for each group of bars
        group_width = (total_bars_per_session * bar_width + gap) * total_session_types

        # Calculate the total width for all groups
        total_width = group_width * total_ticks

        # Define the bar width and gap
        bar_width = 0.8  # Increase this to make the bars wider
        gap = 2.0  # Increase this to make the gap between groups of bars larger

        # Initialize the starting x position
        x_pos = np.arange(
            len(selected_names) * total_session_types * total_bars_per_session + len(selected_names) * gap)

        # Loop over the selected names
        for i, name in enumerate(selected_names):
            # Get the data for the current name
            data = self.get_data(name)

            # Loop over the session types
            for j, session_type in enumerate([0, 1]):
                # Loop over the metrics
                for k, metric in enumerate(metrics):
                    # Calculate the x position for the current bar
                    bar_x_pos = int(
                        i * (total_session_types * total_bars_per_session + gap) + j * total_bars_per_session + k)

                    # Get the data values for the current session type and metric
                    values = data[metric][j]

                    # Plot the bars for the current metric and session type
                    self.ax.bar(x_pos[bar_x_pos], values, color=colors[k], edgecolor=edge_colors[j], width=bar_width)

        # Set the x-ticks to correspond to the selected_names
        self.ax.set_xticks(np.arange(len(selected_names)) * (
                    total_session_types * total_bars_per_session + gap) + total_bars_per_session)

        # Set the x-tick labels to be the numbers corresponding to the selected names
        self.ax.set_xticklabels([self.name_to_number[name] for name in selected_names], rotation=90)

        # Create the legend for the metrics
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=metric) for color, metric in zip(colors, metrics)]
        legend1 = self.ax.legend(handles=legend_elements, loc='upper left')

        # Create the legend for the session types
        session_legend_elements = [Patch(facecolor='white', edgecolor=color, label=f'Session Type {i}') for i, color
                                   in enumerate(edge_colors)]
        legend2 = self.ax.legend(handles=session_legend_elements, loc='upper right')

        # Add the first legend as an artist to the plot
        self.ax.add_artist(legend1)

        # Redraw the canvas
        self.canvas.draw()

    def go_back(self):
        if self.callback:
            self.callback()
