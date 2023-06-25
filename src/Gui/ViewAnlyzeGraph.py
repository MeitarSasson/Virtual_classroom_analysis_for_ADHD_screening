import os
import tkinter

import couchdb
import customtkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from tkinter import ttk, messagebox

from backend import Model
from main import PaddedFrame, BasePage


class viewAnlyzeGraph(PaddedFrame):
    def __init__(self, parent, parent_frame, student_data=None, dataframe_data=None, callback=None, file=None,
                 **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.callback = callback
        self.parent = parent
        self.file = file  # Assign the file parameter to the instance variable
        self.student_data = student_data  # Assign the student_data parameter to the instance variable
        self.dataframe_data = dataframe_data  # Assign the dataframe_data parameter to the instance variable

        # Connect to the CouchDB server
        couch = couchdb.Server('http://admin:1234@localhost:5984')

        # Select the database
        db = couch['test_dataframes']

        # Fetch all documents from the database
        rows = db.view('_all_docs', include_docs=True)

        # Extract the document data from each row
        data = [row['doc'] for row in rows]

        # Convert each document into a DataFrame and concatenate them
        dfs = []
        for doc in data:
            df = pd.DataFrame(doc)  # Convert the document into a DataFrame
            dfs.append(df)
        self.df = pd.concat(dfs)
        print("df = ", self.df)

        self.label_2 = ttk.Label(master=self, font=("Helvetica", 30), text="View Graphs", justify=tkinter.LEFT)
        self.label_2.grid(row=0, column=0, pady=10, padx=10)
        self.model = Model()  # Initialize the model
        if self.file:
            self.button_1 = ttk.Button(master=self, text="Predict", command=self.predict)  # Use ttk.Button
            self.button_1.grid(row=2, column=0)
        # Load the JSON file
        self.data = self.load_json(self.file)

        # Create a combobox for plot selection
        self.plot_type = tkinter.StringVar()
        self.plot_combobox = ttk.Combobox(master=self, textvariable=self.plot_type)
        self.plot_combobox['values'] = ('3D Scatter', 'Bar Plot', 'Results')  # Add 'Results' option
        self.plot_combobox.grid(row=1, column=0)
        self.plot_combobox.current(0)  # set initial selection
        self.plot_combobox.bind("<<ComboboxSelected>>", self.update_plot)
        # Create a Treeview widget for displaying the results
        self.tree = ttk.Treeview(master=self)
        self.tree.grid(row=4, column=0, sticky='nsew')
        # Create a Text widget for displaying the results
        # self.results_text = tkinter.Text(master=self)
        # self.results_text.grid(row=4, column=0, sticky='nsew')
        # Create the plots
        self.create_plots()

        self.button_2 = ttk.Button(master=self, text="Back", command=self.back)
        self.button_2.grid(row=3, column=0)

    def load_json(self, doc):
        if doc is None and self.student_data is not None:
            return self.student_data['adhd_test_details']
        else:
            return doc  # The document is already a dictionary, so just return it

    def predict(self):
        doc = self.file
        y_pred, y_pred_proba = self.model.predict_single(doc)
        # Display the prediction in a message box
        messagebox.showinfo("Prediction Result", f"Prediction: {y_pred}\nPrediction probability: {y_pred_proba}")

    def create_plots(self):
        # Create a new matplotlib figure
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)

        # Create a canvas and add it to the tkinter frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=5, column=0, sticky='nsew')

        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Initial plot
        self.update_plot()

    def update_plot(self, event=None):
        # Clear the current plot
        self.fig.clear()

        if self.plot_type.get() == 'Results':
            # Get the individual's name from the JSON data
            individual_name = self.data['Individual']['Name']
            print(f"Individual's name: {individual_name}")  # Print the individual's name for debugging

            # Filter the DataFrame to get the records for the individual
            individual_records = self.df[self.df['Name'] == individual_name]
            print(f"Individual's records:\n{individual_records}")  # Print the records for debugging

            # Select only the columns you're interested in
            columns_to_display = ['ADHD', 'Medication', 'Response Accuracy', 'Commission Error',
                                  'Omission Error', 'Mean Delay', 'Average Vector Size', 'Session Type']
            individual_records = individual_records[columns_to_display]

            # Remove duplicates
            individual_records = individual_records.drop_duplicates()

            # Define the column names
            self.tree["columns"] = columns_to_display

            # For each column
            for col in columns_to_display:
                # Create the column in the treeview
                self.tree.column(col, width=100)
                # Create the heading for the column
                self.tree.heading(col, text=col)

            # Clear the treeview
            for i in self.tree.get_children():
                self.tree.delete(i)

            # For each row in the DataFrame
            for index, row in individual_records.iterrows():
                # Add the row to the treeview
                self.tree.insert('', 'end', values=list(row))

            # Grid the treeview widget
            self.tree.grid(row=4, column=0, sticky='nsew')
        else:
            # Hide the treeview widget
            self.tree.grid_remove()
        print("data = ", self.data)
        colors = ['blue', 'red']  # Define two colors for the two classes
        class_labels = self.data['Individual']['Type'].keys()  # Get the class labels

        # Map the session type labels to their corresponding integer values
        session_type_map = {'SessionWithoutDisturbances': 0, 'SessionWithDisturbances': 1}

        if self.plot_type.get() == '3D Scatter':
            ax = self.fig.add_subplot(111, projection='3d')
            for i, class_label in enumerate(class_labels):
                # Extract the head rotation data for the current class
                head_rotation = self.data['Individual']['Type'][class_label]['HeadRotation']

                # Separate the x, y, and z coordinates
                x = [point['x'] for point in head_rotation]
                y = [point['y'] for point in head_rotation]
                z = [point['z'] for point in head_rotation]

                # Create a 3D scatter plot with the color corresponding to the current class
                ax.scatter(x, y, z, color=colors[i], label=class_label)  # Add label to the scatter plot

            ax.legend()

        elif self.plot_type.get() == 'Bar Plot':
            # Define the bar width
            bar_width = 0.25  # Adjust the bar width to fit the additional bar

            ax = self.fig.add_subplot(111)
            for i, class_label in enumerate(class_labels):
                # Extract the sizes of the lists
                pressed_and_should = len(self.data['Individual']['Type'][class_label]['PressedAndShould'])
                pressed_and_should_not = len(self.data['Individual']['Type'][class_label]['PressedAndShouldNot'])
                not_pressed_and_should = len(self.data['Individual']['Type'][class_label]['NotPressedAndShould'])

                # Loop over the filenames
                for j, filename in enumerate(self.df['Filename'].unique()):
                    # Extract the ADHD label and the postfix from the filename
                    adhd_label, postfix = filename.split('_')

                    # Convert the ADHD label and the postfix to integers
                    adhd_label = int(adhd_label)
                    postfix = int(postfix)

                    # Filter the DataFrame
                    mean_delay = self.df.loc[(self.df['Session Type'] == session_type_map[class_label]) & (
                            self.df['Filename'] == filename) & (self.df['ADHD'] == adhd_label), 'Mean Delay'].values[0]

                    # Create a bar plot with the bars offset by the bar width times the class index
                    ax.bar(np.arange(4) + i * bar_width,
                           [pressed_and_should, pressed_and_should_not, not_pressed_and_should, mean_delay],
                           width=bar_width, color=colors[i],
                           label=class_label if j == 0 else "")  # Add label to the bar plot

            ax.legend()

            # Set the x-ticks to be the middle of the three bars
            ax.set_xticks(np.arange(4) + bar_width)

            # Set the x-tick labels
            ax.set_xticklabels(['Response Accuracy', 'Commission Error', 'Omission Error', 'Mean Delay'])

        # Redraw the canvas
        self.canvas.draw()

    def back(self):
        if self.callback:
            self.callback()
